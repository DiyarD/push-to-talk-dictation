//! Dictation hotkey: Win+Shift+D toggles record -> transcribe -> paste.
//!
//! The UI is the orb (see `orb.rs`): mic open shows it idle, voice flips it to
//! listening, the click or the hotkey stops the turn, and it thinks until the
//! text is pasted. Clicking a finished orb copies the text instead.
//!
//! Idle cost: this exe only (~MBs). The CUDA worker (win_serve.py) is spawned
//! on first press and killed after IDLE_SECS of disuse -> zero idle memory.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod orb;

use std::cell::RefCell;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use orb::Phase;
use windows::Win32::Foundation::{HANDLE, HGLOBAL, HWND};
use windows::Win32::System::DataExchange::*;
use windows::Win32::System::Memory::*;
use windows::Win32::System::Ole::CF_UNICODETEXT;
use windows::Win32::UI::Input::KeyboardAndMouse::*;
use windows::Win32::UI::WindowsAndMessaging::*;

// ---------------------------------------------------------------- config
const HOTKEY_ID: i32 = 1;
const HOTKEY_MOD: HOT_KEY_MODIFIERS = HOT_KEY_MODIFIERS(MOD_WIN.0 | MOD_SHIFT.0);
const HOTKEY_VK: u32 = 0x44; // 'D'
// bare Escape, registered only while a turn is in flight, released at stop.
const HOTKEY_ESC_ID: i32 = 2;
const PORT: u16 = 8765;
const IDLE_SECS: u64 = 180;
const CREATE_NO_WINDOW: u32 = 0x0800_0000;
// Fallback frame clock, only if the orb window could not be created.
const TIMER_HEADLESS: usize = 8;

fn worker_dir() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let debug_or_release = exe.parent().unwrap();
    // target/debug/hotkey.exe -> C:\dev\dictation\hotkey, worker is ..\worker
    if (debug_or_release.ends_with("debug") || debug_or_release.ends_with("release"))
        && let Some(p) = debug_or_release.parent().and_then(|p| p.parent())
    {
        let w = p.join("worker");
        if w.join("win_serve.py").exists() {
            return w;
        }
    }
    // installed next to worker/
    if exe
        .parent()
        .unwrap()
        .join("worker")
        .join("win_serve.py")
        .exists()
    {
        return exe.parent().unwrap().join("worker");
    }
    PathBuf::from(r"C:\dev\dictation\worker")
}

fn model_dir() -> PathBuf {
    worker_dir()
        .parent()
        .unwrap()
        .join("models")
        .join("phonon-2")
}

fn python_exe() -> String {
    if let Ok(p) = std::env::var("DICTATION_PYTHON") {
        return p;
    }
    for base in [
        std::env::var("USERPROFILE").unwrap_or_default() + r"\.conda\envs\dictation",
        r"C:\ProgramData\anaconda3\envs\dictation".to_string(),
    ] {
        let p = PathBuf::from(&base).join("python.exe");
        if p.exists() {
            return p.to_string_lossy().into_owned();
        }
    }
    "python".to_string()
}

// ---------------------------------------------------------------- state
struct Recorder {
    _stream: cpal::Stream,
    buf: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
    channels: u16,
}

/// What the background turn produced. Handed to the UI thread by `pump_turn`.
enum TurnOutcome {
    Text(String),
    Empty,
    Failed(String),
    Aborted,
}

struct App {
    server: Option<Child>,
    server_ready: bool,
    load_error: Option<String>,
    last_used: Instant,
    busy: bool, // a recording/transcription/load is in flight (watchdog must not kill)
    outcome: Option<TurnOutcome>,
}

impl App {
    fn new() -> Self {
        Self {
            server: None,
            server_ready: false,
            load_error: None,
            last_used: Instant::now() - Duration::from_secs(IDLE_SECS + 1),
            busy: false,
            outcome: None,
        }
    }
}

fn app() -> &'static Mutex<App> {
    static CELL: OnceLock<Mutex<App>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(App::new()))
}

/// True once the worker's model is in memory: the orb crossfades to deep blue.
fn model_ready() -> bool {
    app().lock().unwrap().server_ready
}

// Recording state lives on the main thread only: cpal::Stream is !Send.
thread_local! {
    static RECORDING: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

// Set by Esc or by clicking a Thinking orb: discard the result when it returns.
static ABORT: AtomicBool = AtomicBool::new(false);

// last COMPLETED transcription: the only thing Copy ever copies, so status
// lines like "pasted (N chars)" never leak into the clipboard.
fn copy_text() -> &'static Mutex<String> {
    static CELL: OnceLock<Mutex<String>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(String::new()))
}

fn null_hwnd() -> HWND {
    HWND(std::ptr::null_mut())
}

fn log_path() -> PathBuf {
    worker_dir().join("hotkey.log")
}

fn log_line(msg: &str) {
    use std::io::Write as _;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path())
    {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = writeln!(f, "[{secs}] {msg}");
    }
    println!("[hotkey] {msg}");
}

// ---------------------------------------------------------------- audio
fn start_recording() -> Result<Recorder, String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| "no input device".to_string())?;
    // Prefer exact 16 kHz mono f32; else take default and resample.
    let mut exact: Option<cpal::SupportedStreamConfig> = None;
    if let Ok(configs) = device.supported_input_configs() {
        for range in configs {
            if range.channels() == 1
                && range.sample_format() == cpal::SampleFormat::F32
                && range.min_sample_rate().0 <= 16000
                && range.max_sample_rate().0 >= 16000
            {
                exact = Some(range.with_sample_rate(cpal::SampleRate(16000)));
                break;
            }
        }
    }
    let cfg = match exact {
        Some(c) => c,
        None => device
            .default_input_config()
            .map_err(|e| format!("default input: {e}"))?,
    };
    let (sr, ch) = (cfg.sample_rate().0, cfg.channels());
    let buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::with_capacity(16_000 * 180)));
    let buf2 = buf.clone();
    let stream = device
        .build_input_stream(
            &cfg.into(),
            move |data: &[f32], _| {
                // One extra pass for the level the orb reacts to: a multiply-add
                // per sample and one store per callback, nothing else.
                let sum = data.iter().map(|s| s * s).sum::<f32>();
                orb::push_level((sum / data.len().max(1) as f32).sqrt());
                buf2.lock().unwrap().extend_from_slice(data);
            },
            |err| eprintln!("[hotkey] stream error: {err}"),
            None,
        )
        .map_err(|e| format!("build stream: {e}"))?;
    stream.play().map_err(|e| format!("play: {e}"))?;
    Ok(Recorder {
        _stream: stream,
        buf,
        sample_rate: sr,
        channels: ch,
    })
}

fn to_16k_mono(samples: &[f32], from_rate: u32, channels: u16) -> Vec<f32> {
    if from_rate == 16000 && channels == 1 {
        return samples.to_vec();
    }
    let frames = samples.len() / channels as usize;
    let mut mono = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut s = 0.0f32;
        for c in 0..channels as usize {
            s += samples[f * channels as usize + c];
        }
        mono.push(s / channels as f32);
    }
    if from_rate == 16000 {
        return mono;
    }
    let ratio = 16000.0 / from_rate as f64;
    let n = (mono.len() as f64 * ratio) as usize;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let pos = i as f64 / ratio;
        let j = pos.floor() as usize;
        let frac = (pos - j as f64) as f32;
        let a = mono[j.min(mono.len() - 1)];
        let b = mono[(j + 1).min(mono.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

fn write_wav(path: &Path, samples: &[f32]) -> Result<(), String> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec).map_err(|e| e.to_string())?;
    for s in samples {
        w.write_sample(*s).map_err(|e| e.to_string())?;
    }
    w.finalize().map_err(|e| e.to_string())
}

// ---------------------------------------------------------------- server
fn port_open() -> bool {
    TcpStream::connect(format!("127.0.0.1:{PORT}")).is_ok()
}

fn pid_file() -> PathBuf {
    worker_dir().join("dictation-worker.pid")
}

// A dead hotkey (crash, Task Manager, update restart) leaves its worker
// behind with no owner. The pid file lets the next instance take over.
fn read_stale_pid() -> Option<u32> {
    std::fs::read_to_string(pid_file())
        .ok()?
        .trim()
        .parse()
        .ok()
}

fn pid_is_dictation_python(pid: u32) -> bool {
    let out = Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .creation_flags(CREATE_NO_WINDOW)
        .output();
    match out {
        Ok(o) => {
            let s = String::from_utf8_lossy(&o.stdout).to_lowercase();
            s.contains(&format!("\"{pid}\"")) && s.contains("python.exe")
        }
        Err(_) => false,
    }
}

fn kill_pid(pid: u32) {
    let _ = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/F"])
        .creation_flags(CREATE_NO_WINDOW)
        .output();
}

fn takeover_stale_worker() {
    if let Some(pid) = read_stale_pid() {
        if port_open() && pid_is_dictation_python(pid) {
            log_line(&format!("taking over: killing stale worker pid {pid}"));
            kill_pid(pid);
            std::thread::sleep(Duration::from_secs(2));
            let _ = std::fs::remove_file(pid_file());
        } else if !pid_is_dictation_python(pid) {
            // Recorded pid is dead or is some other program: the file is stale
            // bookkeeping, so clear it. Deliberately left alone while the
            // worker is alive but not yet listening -- that worker owns it,
            // and it may still be mid-load.
            let _ = std::fs::remove_file(pid_file());
        }
    }
}

/// Spawn the CUDA worker if it is not already up. Runs off the UI thread: the
/// orb keeps animating and stays clickable for the whole (cold) load.
fn ensure_server() -> Result<(), String> {
    {
        let st = app().lock().unwrap();
        if st.server.is_some() && port_open() {
            return Ok(());
        }
    }
    // stale child handle?
    {
        let mut st = app().lock().unwrap();
        if let Some(mut child) = st.server.take() {
            let _ = child.kill();
        }
    }
    log_line("spawning worker");
    let wd = worker_dir();
    let py = python_exe();
    let mut cmd = Command::new(&py);
    // Make the child independent of login-shell PATH: conda env layouts keep
    // DLLs in Scripts/ and Library\bin; bare Startup environments lack them.
    if let Some(root) = PathBuf::from(&py).parent() {
        let mut dirs = vec![
            root.join("Scripts"),
            root.join("Library").join("bin"),
            root.to_path_buf(),
        ];
        if let Some(old) = std::env::var_os("PATH") {
            dirs.extend(std::env::split_paths(&old));
        }
        if let Ok(joined) = std::env::join_paths(dirs) {
            cmd.env("PATH", joined);
        }
    }
    // Truncate the worker log each spawn so "see worker log" is always fresh.
    let wlog = std::fs::File::create(wd.join("dictation-worker.log"))
        .map_err(|e| format!("worker log: {e}"))?;
    let wlog2 = wlog.try_clone().map_err(|e| e.to_string())?;
    cmd.current_dir(&wd)
        .arg("-u")
        .arg("win_serve.py")
        .arg("serve")
        .arg("--model-dir")
        .arg(model_dir())
        .arg("--port")
        .arg(PORT.to_string())
        .env("PYTHONNOUSERSITE", "1")
        .env("KMP_DUPLICATE_LIB_OK", "TRUE")
        .stdout(wlog)
        .stderr(wlog2)
        .creation_flags(CREATE_NO_WINDOW);
    let child = cmd.spawn().map_err(|e| format!("spawn worker: {e}"))?;
    let _ = std::fs::write(pid_file(), child.id().to_string());
    app().lock().unwrap().server = Some(child);
    app().lock().unwrap().last_used = Instant::now();
    for _ in 0..750 {
        std::thread::sleep(Duration::from_millis(200));
        if port_open() {
            app().lock().unwrap().last_used = Instant::now();
            log_line("worker ready");
            return Ok(());
        }
        let exited = app()
            .lock()
            .unwrap()
            .server
            .as_mut()
            .map(|c| c.try_wait().ok().flatten().is_some())
            .unwrap_or(true);
        if exited {
            app().lock().unwrap().server = None;
            let _ = std::fs::remove_file(pid_file());
            log_line("worker exited during load");
            return Err("worker exited during load (see worker/dictation-worker.log)".to_string());
        }
    }
    log_line("worker port timeout");
    Err("worker did not open port in 150s (see worker/dictation-worker.log)".to_string())
}

fn watchdog() {
    std::thread::spawn(|| {
        loop {
            std::thread::sleep(Duration::from_secs(15));
            let kill = {
                let st = app().lock().unwrap();
                st.server.is_some()
                    && !st.busy
                    && st.last_used.elapsed() > Duration::from_secs(IDLE_SECS)
            };
            if kill {
                let mut st = app().lock().unwrap();
                if let Some(mut child) = st.server.take() {
                    let _ = child.kill();
                    let _ = std::fs::remove_file(pid_file());
                    log_line("worker idle-evicted");
                }
            }
        }
    });
}

// ---------------------------------------------------------------- http
fn post_wav(wav: &Path) -> Result<String, String> {
    let data = std::fs::read(wav).map_err(|e| e.to_string())?;
    let boundary = "----dictation7MA4YWxkTrZu0G";
    let mut body: Vec<u8> = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\n",
    );
    body.extend_from_slice(&data);
    body.extend_from_slice(format!("\r\n--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\nphonon-2");
    body.extend_from_slice(format!("\r\n--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"response_format\"\r\n\r\nverbose_json",
    );
    body.extend_from_slice(format!("\r\n--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"timestamp_granularities[]\"\r\n\r\nword",
    );
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    let mut sock =
        TcpStream::connect(format!("127.0.0.1:{PORT}")).map_err(|e| format!("connect: {e}"))?;
    sock.set_read_timeout(Some(Duration::from_secs(300))).ok();
    let req = format!(
        "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: multipart/form-data; boundary={boundary}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    sock.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    sock.write_all(&body).map_err(|e| e.to_string())?;
    let mut resp: Vec<u8> = Vec::new();
    sock.read_to_end(&mut resp).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&resp);
    let (head, json_body) = text.split_once("\r\n\r\n").ok_or("bad http reply")?;
    if !head.contains(" 200 ") {
        return Err(format!(
            "server: {}",
            json_body.chars().take(300).collect::<String>()
        ));
    }
    let v: serde_json::Value = serde_json::from_str(json_body).map_err(|e| e.to_string())?;
    if let Some(t) = v.get("text").and_then(|t| t.as_str()) {
        let words: Vec<(String, f64, f64)> = v
            .get("words")
            .and_then(|w| w.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|w| {
                        Some((
                            w.get("word")?.as_str()?.to_string(),
                            w.get("start")?.as_f64()?,
                            w.get("end")?.as_f64()?,
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let walls = read_walls(wav);
        return Ok(cleanup(&with_pause_marks(t, &words, &walls)));
    }
    Err(format!(
        "no text: {}",
        json_body.chars().take(200).collect::<String>()
    ))
}

// Gaps of PAUSE_S+ seconds between words become "..." so a reader (human or
// LLM) can see where you paused. Two sources, unioned: decoder word gaps,
// plus wall-clock silences anchored to the first word after their midpoint
// (TDT durations smear quiet pauses into stretched words, hiding the gap).
// Verified against the engine text — on any mismatch it wins unchanged.
const PAUSE_S: f64 = 1.5;
fn wall_silences(pcm: &[f32]) -> Vec<(f32, f32)> {
    const BLOCK: usize = 800; // 50 ms @ 16 kHz
    if pcm.len() < BLOCK {
        return vec![];
    }
    let n = pcm.len() / BLOCK;
    let mut rms = Vec::with_capacity(n);
    let mut peak = 0.0f32;
    for i in 0..n {
        let s = &pcm[i * BLOCK..(i + 1) * BLOCK];
        let r = (s.iter().map(|x| x * x).sum::<f32>() / BLOCK as f32).sqrt();
        peak = peak.max(r);
        rms.push(r);
    }
    let gate = 0.004f32.max(0.18 * peak);
    let mut out = vec![];
    let mut st: Option<usize> = None;
    for (i, &r) in rms.iter().enumerate() {
        if r <= gate && st.is_none() {
            st = Some(i);
        } else if r > gate && st.is_some() {
            let a = st.take().unwrap();
            if (i - a) as f32 * 0.05 >= PAUSE_S as f32 {
                out.push((a as f32 * 0.05, i as f32 * 0.05));
            }
        }
    }
    if let Some(a) = st
        && (n - a) as f32 * 0.05 >= PAUSE_S as f32
    {
        out.push((a as f32 * 0.05, n as f32 * 0.05));
    }
    out
}

fn read_walls(path: &Path) -> Vec<(f32, f32)> {
    match hound::WavReader::open(path) {
        Ok(mut r) => {
            let pcm: Vec<f32> = r.samples::<f32>().filter_map(|s| s.ok()).collect();
            wall_silences(&pcm)
        }
        Err(_) => vec![],
    }
}

fn with_pause_marks(text: &str, words: &[(String, f64, f64)], walls: &[(f32, f32)]) -> String {
    if words.is_empty() {
        return text.to_string();
    }
    let mut mark = vec![false; words.len()];
    let mut prev_end = 0.0;
    for (i, (_, s, e)) in words.iter().enumerate() {
        if i > 0 && *s - prev_end >= PAUSE_S {
            mark[i] = true;
        }
        prev_end = *e;
    }
    for (s, e) in walls {
        let mid = (*s as f64 + *e as f64) / 2.0;
        if let Some(idx) = words.iter().position(|(_, st, _)| *st >= mid)
            && idx > 0
        {
            mark[idx] = true;
        }
    }
    let mut out = String::new();
    for (i, (w, _, _)) in words.iter().enumerate() {
        if i > 0 {
            out.push_str(if mark[i] { " ... " } else { " " });
        }
        out.push_str(w);
    }
    let norm = |t: &str| t.split_whitespace().collect::<Vec<_>>().join(" ");
    let normed = norm(&out);
    let stripped: Vec<&str> = normed.split(' ').filter(|t| *t != "...").collect();
    if stripped.join(" ") == norm(text) {
        out
    } else {
        text.to_string()
    }
}

// ---------------------------------------------------------------- cleanup (rule-based, deterministic)
const FILLERS: &[&str] = &["um", "umm", "uh", "uhh", "er", "ah", "hmm", "mm-hmm"];
fn cleanup(text: &str) -> String {
    // Shield pause markers from the passes below (space-dot runs would glue).
    let shielded = text.replace(" ... ", " \u{E000} ");
    let mut words: Vec<String> = Vec::new();
    for tok in shielded.split_whitespace() {
        let core = tok
            .trim_matches(|c: char| ",.?!;:\"'()[]".contains(c))
            .to_lowercase();
        if FILLERS.contains(&core.as_str()) {
            continue;
        }
        if let Some(last) = words.last()
            && !core.is_empty()
            && last.to_lowercase() == core
        {
            continue; // collapse "the the"
        }
        words.push(tok.to_string());
    }
    let mut s = words.join(" ");
    for (a, b) in [
        (" ,", ","),
        (" .", "."),
        (" ;", ";"),
        (" :", ":"),
        (" !", "!"),
        (" ?", "?"),
    ] {
        s = s.replace(a, b);
    }
    while s.contains("  ") {
        s = s.replace("  ", " ");
    }
    let mut s = s.trim().to_string();
    s = s.replace('\u{E000}', "...");
    while s.starts_with("... ") {
        s.replace_range(.."... ".len(), "");
    }
    while s.ends_with(" ...") {
        s.truncate(s.len() - " ...".len());
    }
    let mut s = s.trim().to_string();
    if let Some(c) = s.chars().next() {
        let up: String = c.to_uppercase().collect();
        s = up + &s[c.len_utf8()..];
    }
    s
}

// ---------------------------------------------------------------- inject
fn send_unicode(text: &str) {
    use std::mem::size_of;
    let mut inputs: Vec<INPUT> = Vec::new();
    for u in text.encode_utf16() {
        let scan = u;
        inputs.push(INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: windows::Win32::UI::Input::KeyboardAndMouse::INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: VIRTUAL_KEY(0),
                    wScan: scan,
                    dwFlags: KEYEVENTF_UNICODE,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        });
        inputs.push(INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: windows::Win32::UI::Input::KeyboardAndMouse::INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: VIRTUAL_KEY(0),
                    wScan: scan,
                    dwFlags: KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        });
    }
    unsafe {
        SendInput(&inputs, size_of::<INPUT>() as i32);
    }
}

fn key_down_up(vk: VIRTUAL_KEY, up: bool) {
    use std::mem::size_of;
    let ki = KEYBDINPUT {
        wVk: vk,
        wScan: 0,
        dwFlags: if up {
            KEYEVENTF_KEYUP
        } else {
            KEYBD_EVENT_FLAGS(0)
        },
        time: 0,
        dwExtraInfo: 0,
    };
    let input = INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: windows::Win32::UI::Input::KeyboardAndMouse::INPUT_0 { ki },
    };
    unsafe {
        SendInput(&[input], size_of::<INPUT>() as i32);
    }
}

fn get_clip_text() -> Option<Vec<u16>> {
    unsafe {
        if OpenClipboard(null_hwnd()).is_err() {
            return None;
        }
        let out = (|| {
            if IsClipboardFormatAvailable(CF_UNICODETEXT.0 as u32).is_err() {
                return None;
            }
            let h = GetClipboardData(CF_UNICODETEXT.0 as u32).ok()?;
            let ptr = GlobalLock(HGLOBAL(h.0)) as *const u16;
            if ptr.is_null() {
                return None;
            }
            let mut len = 0usize;
            while *ptr.add(len) != 0 {
                len += 1;
            }
            let mut v = vec![0u16; len + 1];
            std::ptr::copy_nonoverlapping(ptr, v.as_mut_ptr(), len + 1);
            let _ = GlobalUnlock(HGLOBAL(h.0));
            Some(v)
        })();
        let _ = CloseClipboard();
        out
    }
}

fn set_clip_text(text: &str) -> bool {
    unsafe {
        if OpenClipboard(null_hwnd()).is_err() {
            return false;
        }
        let ok = (|| {
            if EmptyClipboard().is_err() {
                return None;
            }
            let wide: Vec<u16> = text.encode_utf16().chain([0]).collect();
            let bytes = wide.len() * 2;
            let h = GlobalAlloc(GMEM_MOVEABLE, bytes).ok()?;
            let ptr = GlobalLock(h) as *mut u16;
            if ptr.is_null() {
                return None;
            }
            std::ptr::copy_nonoverlapping(wide.as_ptr(), ptr, wide.len());
            let _ = GlobalUnlock(HGLOBAL(h.0));
            SetClipboardData(CF_UNICODETEXT.0 as u32, HANDLE(h.0)).ok()?;
            Some(())
        })();
        let _ = CloseClipboard();
        ok.is_some()
    }
}

fn paste_text(text: &str) {
    unsafe {
        let saved = get_clip_text();
        if set_clip_text(text) {
            std::thread::sleep(Duration::from_millis(80));
            key_down_up(VK_CONTROL, false);
            key_down_up(VK_V, false);
            key_down_up(VK_V, true);
            key_down_up(VK_CONTROL, true);
            std::thread::sleep(Duration::from_millis(300));
            if OpenClipboard(null_hwnd()).is_ok() {
                let _ = EmptyClipboard();
                if let Some(prev) = saved {
                    let bytes = prev.len() * 2;
                    if let Ok(h) = GlobalAlloc(GMEM_MOVEABLE, bytes) {
                        let p = GlobalLock(h) as *mut u16;
                        if !p.is_null() {
                            std::ptr::copy_nonoverlapping(prev.as_ptr(), p, prev.len());
                            let _ = GlobalUnlock(HGLOBAL(h.0));
                            let _ = SetClipboardData(CF_UNICODETEXT.0 as u32, HANDLE(h.0));
                        }
                    }
                }
                let _ = CloseClipboard();
            }
        } else {
            // clipboard busy: type it directly (slow but works almost everywhere)
            send_unicode(text);
        }
    }
}

fn esc_register() {
    unsafe {
        if RegisterHotKey(
            null_hwnd(),
            HOTKEY_ESC_ID,
            HOT_KEY_MODIFIERS(0),
            VK_ESCAPE.0 as u32,
        )
        .is_err()
        {
            log_line("esc hotkey busy, abort via Esc unavailable this turn");
        }
    }
}

fn esc_unregister() {
    unsafe {
        let _ = UnregisterHotKey(null_hwnd(), HOTKEY_ESC_ID);
    }
}

// ---------------------------------------------------------------- flow
fn tmp_wav() -> PathBuf {
    std::env::temp_dir().join("dictation_last.wav")
}

fn finish_turn(out: TurnOutcome) {
    app().lock().unwrap().outcome = Some(out);
    pump_turn();
}

/// Called on the UI thread once per orb frame: hand a finished turn to the orb.
fn pump_turn() {
    let Some(out) = app().lock().unwrap().outcome.take() else {
        return;
    };
    app().lock().unwrap().busy = false;
    esc_unregister();
    match out {
        TurnOutcome::Text(t) => {
            orb::ready();
            log_line(&format!("pasted {} chars", t.len()));
        }
        TurnOutcome::Empty => {
            log_line("no speech detected");
            orb::fail();
        }
        TurnOutcome::Failed(e) => {
            log_line(&e);
            orb::fail();
        }
        TurnOutcome::Aborted => {
            log_line("turn aborted, result discarded");
            orb::hide();
        }
    }
}

/// Called on the UI thread when the orb is clicked. One verb per phase.
fn on_orb_click(p: Phase) {
    match p {
        Phase::Idle | Phase::Listening => stop_and_transcribe(),
        Phase::Thinking => do_abort(),
        Phase::Ready => {
            let t = copy_text().lock().unwrap().clone();
            orb::copied();
            if t.trim().is_empty() {
                return;
            }
            set_clip_text(&t);
            log_line(&format!("clicked: copied {} chars to clipboard", t.len()));
        }
        Phase::Error => orb::hide(),
    }
}

/// Abort the in-flight turn: a recording is discarded, a pending transcription
/// is dropped when its blocking call returns.
fn do_abort() {
    let had_recording = RECORDING.with(|r| r.borrow_mut().take().is_some());
    if had_recording {
        orb::hide();
        app().lock().unwrap().busy = false;
        log_line("turn aborted (recording discarded)");
        return;
    }
    if app().lock().unwrap().busy {
        ABORT.store(true, Ordering::SeqCst);
        log_line("abort requested");
    }
    // idle: ignore, Esc keeps its normal meaning everywhere
}

fn on_hotkey() {
    if RECORDING.with(|r| r.borrow().is_some()) {
        stop_and_transcribe();
        return;
    }
    if app().lock().unwrap().busy {
        // Mid-turn: the orb is already up in Thinking. Esc or a click cancels.
        return;
    }
    // start: mic FIRST (instant), model loads in parallel behind it.
    match start_recording() {
        Ok(rec) => {
            {
                let mut st = app().lock().unwrap();
                st.busy = true;
                st.load_error = None;
                st.server_ready = false;
                st.outcome = None;
            }
            ABORT.store(false, Ordering::SeqCst);
            orb::set_armed(true);
            esc_register();
            RECORDING.with(|r| {
                *r.borrow_mut() = Some(rec);
            });
            // The model loads while the user speaks; a 60s+ turn hides even a
            // cold 20s load completely.
            std::thread::spawn(|| {
                let res = ensure_server();
                let mut st = app().lock().unwrap();
                match res {
                    Ok(()) => {
                        st.server_ready = true;
                        st.last_used = Instant::now();
                    }
                    Err(e) => {
                        st.load_error = Some(e);
                        st.busy = false;
                    }
                }
            });
        }
        Err(e) => {
            log_line(&format!("mic error: {e}"));
            finish_turn(TurnOutcome::Failed(format!("mic error: {e}")));
        }
    }
}

fn stop_and_transcribe() {
    let Some(rec) = RECORDING.with(|r| r.borrow_mut().take()) else {
        return;
    };
    orb::set_armed(false);
    orb::thinking();
    let Recorder {
        _stream,
        buf,
        sample_rate,
        channels,
    } = rec;
    drop(_stream); // mic off immediately
    let raw = buf.lock().unwrap().clone();
    let pcm = to_16k_mono(&raw, sample_rate, channels);
    if pcm.len() < 1600 {
        finish_turn(TurnOutcome::Failed("too short — nothing recorded".into()));
        return;
    }
    let wav = tmp_wav();
    if let Err(e) = write_wav(&wav, &pcm) {
        finish_turn(TurnOutcome::Failed(format!("wav error: {e}")));
        return;
    }
    {
        let mut st = app().lock().unwrap();
        st.busy = true;
        st.outcome = None;
    }
    // Everything blocking happens here, off the UI thread: the orb has to keep
    // breathing through a cold model load and a long decode.
    std::thread::spawn(move || {
        let out = match run_turn(&wav) {
            TurnOutcome::Text(t) if t.trim().is_empty() => TurnOutcome::Empty,
            TurnOutcome::Text(t) => {
                *copy_text().lock().unwrap() = t.clone();
                paste_text(&t);
                app().lock().unwrap().last_used = Instant::now();
                TurnOutcome::Text(t)
            }
            other => other,
        };
        app().lock().unwrap().outcome = Some(out);
    });
}

fn run_turn(wav: &Path) -> TurnOutcome {
    // Normally the worker finished loading long before a turn ends; if not,
    // wait for it here while the orb sits in Thinking.
    let t0 = Instant::now();
    loop {
        let (ready, err) = {
            let st = app().lock().unwrap();
            (st.server_ready, st.load_error.clone())
        };
        if let Some(e) = err {
            return TurnOutcome::Failed(format!(
                "worker error: {e} (audio kept at {})",
                wav.display()
            ));
        }
        if ready {
            break;
        }
        if ABORT.swap(false, Ordering::SeqCst) {
            return TurnOutcome::Aborted;
        }
        if t0.elapsed() > Duration::from_secs(150) {
            return TurnOutcome::Failed(format!(
                "model load timeout (audio kept at {})",
                wav.display()
            ));
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    if ABORT.swap(false, Ordering::SeqCst) {
        return TurnOutcome::Aborted;
    }
    match post_wav(wav) {
        Ok(t) => TurnOutcome::Text(t),
        Err(e) => TurnOutcome::Failed(format!("transcribe error: {e}")),
    }
}

// ---------------------------------------------------------------- main
fn run_test(wav: &str) {
    println!("[hotkey] test mode: {wav}");
    ensure_server().expect("server");
    match post_wav(Path::new(wav)) {
        Ok(text) => println!("[hotkey] TEXT: {text}"),
        Err(e) => {
            eprintln!("[hotkey] ERROR: {e}");
            std::process::exit(1);
        }
    }
    let mut st = app().lock().unwrap();
    if let Some(mut child) = st.server.take() {
        let _ = child.kill();
        let _ = std::fs::remove_file(pid_file());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() >= 3 && args[1] == "--test" {
        run_test(&args[2]);
        return;
    }
    if args.len() >= 3 && args[1] == "--mic-test" {
        let secs: u64 = args[2].parse().unwrap_or(3);
        println!("[hotkey] recording {secs}s from default mic…");
        let rec = start_recording().expect("mic");
        let (sr, ch) = (rec.sample_rate, rec.channels);
        std::thread::sleep(Duration::from_secs(secs));
        let raw = rec.buf.lock().unwrap().clone();
        drop(rec);
        let pcm = to_16k_mono(&raw, sr, ch);
        let peak = pcm.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        let rms = (pcm.iter().map(|s| s * s).sum::<f32>() / pcm.len().max(1) as f32).sqrt();
        println!("[hotkey] samples={} peak={peak:.3} rms={rms:.4}", pcm.len());
        return;
    }
    log_line(&format!(
        "hotkey starting, worker: {}",
        worker_dir().display()
    ));
    // Before any window exists: otherwise the orb's DIB is sized from the
    // virtualised 96 DPI and gets stretched by the compositor on scaled displays.
    if !orb::dpi_aware() {
        log_line("DPI awareness could not be raised; orb may render soft when scaled");
    }
    log_line(&orb::dpi_report());
    // Claim the hotkey before touching anything else: it is the only real
    // single-instance mutex. A duplicate launch must exit *before* it starts
    // adopting workers, otherwise it kills the live instance's worker and
    // deletes its pid file, and the running instance can no longer take over
    // (or detect) that worker on a later restart.
    unsafe {
        if RegisterHotKey(null_hwnd(), HOTKEY_ID, HOTKEY_MOD, HOTKEY_VK).is_err() {
            eprintln!("[hotkey] RegisterHotKey failed (already running?)");
            std::process::exit(1);
        }
    }
    takeover_stale_worker();
    watchdog();
    orb::set_hooks(pump_turn, on_orb_click);
    let have_orb = orb::ensure();
    if !have_orb {
        // No orb: dictation still has to work, so drain finished turns from a
        // plain thread timer instead of the orb's frame clock.
        log_line("orb window could not be created; running without UI");
        unsafe {
            let _ = SetTimer(null_hwnd(), TIMER_HEADLESS, 250, None);
        }
    } else {
        log_line(&orb::geometry_report());
    }
    unsafe {
        let mut msg = MSG::default();
        while GetMessageW(&mut msg, null_hwnd(), 0, 0).as_bool() {
            if msg.message == WM_HOTKEY {
                if msg.wParam.0 as i32 == HOTKEY_ID {
                    on_hotkey();
                } else if msg.wParam.0 as i32 == HOTKEY_ESC_ID {
                    do_abort();
                }
            } else if !have_orb && msg.message == WM_TIMER {
                pump_turn();
            }
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        let _ = UnregisterHotKey(null_hwnd(), HOTKEY_ID);
    }
    orb::destroy();
}
