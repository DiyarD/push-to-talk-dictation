//! The orb: the whole UI.
//!
//! ATTRIBUTION: the dot-matrix animation below is a port of the `MatrixOrb`
//! component from Rare UI (https://rareui.com), Copyright (c) 2026 Swami
//! Malode, licensed under MIT + Commons Clause v1.0 + mandatory attribution.
//! The full licence text is in LICENSE-rare-ui at the repository root and the
//! credit requirements are documented in NOTICE.md. That licence requires this
//! notice to stay in the copied source -- do not remove it.
//!
//! One layered, always-on-top popup whose entire client area is a 32-bit
//! premultiplied DIB. Dots are rasterised analytically into that DIB (no
//! GDI+ / Direct2D) and pushed with UpdateLayeredWindow, so the window has no
//! chrome, no background and no child controls at all.
//!
//! The animation is a port of the Rare UI `MatrixOrb` component: an 11x11 dot
//! matrix whose per-dot radius is driven by a blend of per-state "intensity"
//! functions, smoothed by an asymmetric envelope follower, with a damped
//! spring on the overall scale.

use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{
    COLORREF, HINSTANCE, HWND, LPARAM, LRESULT, POINT, RECT, SIZE, WPARAM,
};
use windows::Win32::Graphics::Gdi::{
    AC_SRC_ALPHA, AC_SRC_OVER, BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BLENDFUNCTION,
    CreateCompatibleDC, CreateDIBSection, DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC,
    GetMonitorInfoW, HDC, HGDIOBJ, MONITOR_DEFAULTTONEAREST, MONITORINFO, MonitorFromWindow,
    ReleaseDC, SelectObject,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, DPI_AWARENESS_CONTEXT_SYSTEM_AWARE,
    DPI_AWARENESS_PER_MONITOR_AWARE, DPI_AWARENESS_SYSTEM_AWARE,
    GetAwarenessFromDpiAwarenessContext, GetDpiForWindow, GetThreadDpiAwarenessContext,
    PROCESS_PER_MONITOR_DPI_AWARE, SetProcessDpiAwareness, SetProcessDpiAwarenessContext,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, GetSystemMetrics, KillTimer, MA_NOACTIVATE, RegisterClassW,
    SM_CXSCREEN, SM_CYSCREEN, SPI_GETCLIENTAREAANIMATION, SPI_GETWORKAREA, SW_HIDE,
    SW_SHOWNOACTIVATE, SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS, SetTimer, ShowWindow,
    SystemParametersInfoW, ULW_ALPHA, UpdateLayeredWindow, WM_DPICHANGED, WM_LBUTTONDOWN,
    WM_MOUSEACTIVATE, WM_TIMER, WNDCLASSW, WS_EX_LAYERED, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW,
    WS_EX_TOPMOST, WS_POPUP,
};
use windows::core::{PCWSTR, w};

// ------------------------------------------------------------------ shape
/// Logical (DPI-independent) side of the square window.
pub const ORB_PX: i32 = 184;
const GRID: usize = 11;
/// Rasterise the matrix at this multiple of the window size and box-filter
/// down, so dot edges land on a quarter-pixel grid instead of smearing.
const SS: f32 = 2.0;
/// 1.12, not the square's 1.41 corner, is what reads as a circle.
const OUTLINE: f32 = 1.12;
/// Fraction of the box the dot *centres* span; the rest is breathing room.
const SPREAD: f32 = 0.80;
/// Dot radius as a fraction of the dot spacing. Kept well under 0.5 so dots
/// stay visibly separate: touching dots merge into one soft blob and the
/// matrix stops reading as a matrix.
const DOT_R: f32 = 0.44;

const MARGIN: i32 = 16;
const TIMER_ID: usize = 7;
const TIMER_MS: u32 = 16;
/// Seconds of dissolve once a phase's life is over.
const FADE: f32 = 0.5;
/// How long a finished orb stays clickable before dissolving.
const READY_LIFE: f32 = 5.0;
const ERROR_LIFE: f32 = 5.0;

// ------------------------------------------------------------------ motion
// Identical to the reference component: the spring is tuned in seconds.
const STIFFNESS: f32 = 180.0;
const DAMPING: f32 = 26.0;
const ATTACK: f32 = 0.22;
const RELEASE: f32 = 0.08;
const BLEND: f32 = 0.16;

const R_IDLE: usize = 0;
const R_LISTEN: usize = 1;
const R_THINK: usize = 2;
const R_READY: usize = 3;
const R_ERROR: usize = 4;
const RSTATES: usize = 5;

/// Spring rest length per render state: the orb swells when you speak.
/// Ready rests at the idle length, because that is what it looks like.
const SCALE: [f32; RSTATES] = [0.88, 1.00, 0.92, 0.88, 0.84];
/// (radius, angular speed, phase, gaussian spread)
const ORBITERS: [(f32, f32, f32, f32); 3] = [
    (0.62, 2.20, 0.0, 0.42),
    (0.40, -1.70, 2.1, 0.36),
    (0.80, 1.15, 4.0, 0.34),
];

// Model loading -> loaded. Same hue, two stops: it reads as warming up.
const C_LIGHT: (f32, f32, f32) = (0.376, 0.647, 0.980); // #60A5FA
const C_DEEP: (f32, f32, f32) = (0.145, 0.388, 0.922); // #2563EB
const C_WARN: (f32, f32, f32) = (0.973, 0.443, 0.443); // #F87171

/// Dot colour for a phase. `ready_mix` is 0 while the model is loading (light
/// blue) and 1 once it is resident (deeper blue); every phase shares that one
/// crossfade, so "which colour" tracks the model and never the phase.
fn colour_of(phase: Phase, ready_mix: f32) -> (f32, f32, f32) {
    if phase == Phase::Error {
        return C_WARN;
    }
    let m = ready_mix;
    (
        C_LIGHT.0 + (C_DEEP.0 - C_LIGHT.0) * m,
        C_LIGHT.1 + (C_DEEP.1 - C_LIGHT.1) * m,
        C_LIGHT.2 + (C_DEEP.2 - C_LIGHT.2) * m,
    )
}

// ------------------------------------------------------------------ audio
/// Latest capture-block RMS. One relaxed store per callback (~10 ms) is the
/// entire cost of voice detection.
static RMS: AtomicU32 = AtomicU32::new(0);

/// Called from the cpal capture callback. O(1), not per-sample work.
pub fn push_level(rms: f32) {
    RMS.store(rms.to_bits(), Ordering::Relaxed);
}

/// Drop the last published level. Called when the mic closes: the capture
/// callback stops, so without this the final RMS would sit in the static
/// forever and every post-recording phase would render at speaking amplitude.
pub fn clear_level() {
    RMS.store(0f32.to_bits(), Ordering::Relaxed);
}

// ------------------------------------------------------------------ state
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// Mic open, nobody has spoken yet.
    Idle,
    /// Mic open, voice above the noise floor.
    Listening,
    /// Waiting on the model and/or transcribing.
    Thinking,
    /// Text is final and has been pasted; clicking copies it.
    Ready,
    /// Something went wrong; dissolves on its own.
    Error,
}

impl Phase {
    fn render(self) -> usize {
        match self {
            Phase::Idle => R_IDLE,
            Phase::Listening => R_LISTEN,
            Phase::Thinking => R_THINK,
            Phase::Ready => R_READY,
            Phase::Error => R_ERROR,
        }
    }

    /// Seconds this phase stays up before dissolving. Armed states never do.
    fn life(self) -> Option<f32> {
        match self {
            Phase::Ready => Some(READY_LIFE),
            Phase::Error => Some(ERROR_LIFE),
            _ => None,
        }
    }
}

struct Hooks {
    /// Called once per frame before rendering, with no lock held.
    tick: fn(),
    /// Called on a click inside the orb, with no lock held.
    click: fn(Phase),
}

/// Leaked once at startup so `&'static Hooks` is sound; null means "no hooks",
/// which is what the orb runs with until the app installs them.
static HOOKS: AtomicPtr<Hooks> = AtomicPtr::new(std::ptr::null_mut());

fn hooks() -> Option<&'static Hooks> {
    let p = HOOKS.load(Ordering::Relaxed);
    (!p.is_null()).then(|| unsafe { &*p })
}

/// Install the two callbacks. Call once, from the thread that owns the window.
pub fn set_hooks(tick: fn(), click: fn(Phase)) {
    let b = Box::into_raw(Box::new(Hooks { tick, click }));
    let old = HOOKS.swap(b, Ordering::Relaxed);
    assert!(old.is_null(), "orb hooks already installed");
}

struct Orb {
    hwnd: usize,
    hdc: usize,
    old_obj: usize,
    bmp: usize,
    bits: usize,
    w: i32,
    h: i32,
    /// Static radial halo falloff, 0..1.
    glow: Vec<f32>,
    /// Dot coverage scratch, 0..1, refilled every frame.
    cov: Vec<f32>,
    /// Supersampled dot coverage, SS*w square.
    hi: Vec<f32>,

    t: f32,
    amp: f32,
    scale: f32,
    vel: f32,
    flash: f32,
    weights: [f32; RSTATES],
    ready_mix: f32,
    opacity: f32,

    phase: Phase,
    armed: bool,
    visible: bool,
    /// Life countdown origin; `phase.life()` is measured from here.
    deadline_start: Instant,
    deadline: Option<Instant>,

    rms: f32,
    floor: f32,
    gate: f32,
    speaking: bool,
    level: f32,

    reduce: bool,
    last: Instant,
    /// Frame signature, so reduced-motion mode only redraws on real changes.
    sig: u64,
}

fn orb() -> &'static Mutex<Option<Orb>> {
    static CELL: OnceLock<Mutex<Option<Orb>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(None))
}

fn hwnd_of(v: usize) -> HWND {
    HWND(v as *mut std::ffi::c_void)
}

fn lparam_xy(lp: LPARAM) -> (i32, i32) {
    (
        ((lp.0 & 0xFFFF) as u16 as i16) as i32,
        (((lp.0 >> 16) & 0xFFFF) as u16 as i16) as i32,
    )
}

fn reduce_motion() -> bool {
    let mut on = 0i32;
    let failed = unsafe {
        SystemParametersInfoW(
            SPI_GETCLIENTAREAANIMATION,
            0,
            Some(&mut on as *mut i32 as *mut std::ffi::c_void),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        )
        .is_err()
    };
    failed || on == 0
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain([0]).collect()
}

// ------------------------------------------------------------------ intensity
fn intensity_of(s: usize, d: f32, nx: f32, ny: f32, t: f32, amp: f32, flash: f32) -> f32 {
    match s {
        R_LISTEN => {
            let ripple = 0.5 + 0.5 * (d * 4.2 - t * 3.0).sin();
            0.32 + amp * (0.34 + 0.38 * ripple)
        }
        R_THINK => {
            let mut heat = 0.0f32;
            for (radius, speed, phase, spread) in ORBITERS {
                let a = t * speed + phase;
                let dx = nx - a.cos() * radius;
                let dy = ny - a.sin() * radius;
                heat += (-(dx * dx + dy * dy) / (spread * spread)).exp();
            }
            0.26 + 0.8 * heat.min(1.0)
        }
        R_READY => {
            // Per spec: once the text is pasted the orb settles back to its
            // resting idle look. `flash` is the only difference - a short pulse
            // on paste and on click - so the moment is acknowledged without
            // inventing a look the user did not ask for.
            0.30 + 0.62 * amp * (0.75 + 0.25 * (t * 1.05 - d * 2.4).sin()) + 0.50 * flash
        }
        R_ERROR => {
            // Uneven and restless, never a clean pulse you can lock onto.
            let j = 0.5 + 0.5 * (t * 6.5 + d * 5.0).sin();
            0.30 + 0.44 * j
        }
        _ => 0.30 + 0.62 * amp * (0.75 + 0.25 * (t * 1.05 - d * 2.4).sin()),
    }
}

/// Analytic coverage of one disc into `cov` (max-blended, one-pixel feather).
fn splat(cov: &mut [f32], w: i32, h: i32, cx: f32, cy: f32, r: f32) {
    let x0 = ((cx - r - 1.0).floor() as i32).clamp(0, w - 1);
    let x1 = ((cx + r + 1.0).ceil() as i32).clamp(0, w - 1);
    let y0 = ((cy - r - 1.0).floor() as i32).clamp(0, h - 1);
    let y1 = ((cy + r + 1.0).ceil() as i32).clamp(0, h - 1);
    for y in y0..=y1 {
        let dy = y as f32 + 0.5 - cy;
        let row = (y * w) as usize;
        for x in x0..=x1 {
            let dx = x as f32 + 0.5 - cx;
            let a = (r - (dx * dx + dy * dy).sqrt() + 0.5).clamp(0.0, 1.0);
            let i = row + x as usize;
            if a > cov[i] {
                cov[i] = a;
            }
        }
    }
}

/// Everything `rasterize` needs about how the orb currently looks.
struct Look {
    t: f32,
    amp: f32,
    flash: f32,
    scale: f32,
    weights: [f32; RSTATES],
}

/// Rasterise the dot matrix at SS times the window size, then box-filter down
/// to `w * w`. Coverage is max-blended per dot, so dots never double-expose.
fn rasterize(cov: &mut [f32], hi: &mut [f32], w: i32, look: &Look) {
    let Look {
        t,
        amp,
        flash,
        scale,
        weights,
    } = *look;
    let hw = (w as f32 * SS).round() as i32;
    let k = hw as f32 / ORB_PX as f32;
    let center = hw as f32 * 0.5;
    let half = (GRID - 1) as f32 * 0.5;
    let spacing = ORB_PX as f32 * SPREAD / (GRID - 1) as f32;
    let max_radius = spacing * DOT_R * k;

    hi.fill(0.0);
    for iy in 0..GRID {
        for ix in 0..GRID {
            let nx = (ix as f32 - half) / half;
            let ny = (iy as f32 - half) / half;
            let d = (nx * nx + ny * ny).sqrt();
            if d > OUTLINE {
                continue;
            }
            let mut blended = 0.0f32;
            for (s, ws) in weights.iter().enumerate() {
                if *ws > 0.001 {
                    blended += *ws * intensity_of(s, d, nx, ny, t, amp, flash);
                }
            }
            let radius = max_radius * (-d * d * 1.7).exp() * blended.clamp(0.0, 1.0) * scale;
            if radius * SS < 0.5 {
                continue;
            }
            splat(
                hi,
                hw,
                hw,
                center + (ix as f32 - half) * spacing * scale * k,
                center + (iy as f32 - half) * spacing * scale * k,
                radius,
            );
        }
    }

    let s = SS as usize;
    let hw = hw as usize;
    let inv = 1.0 / (s * s) as f32;
    for y in 0..w as usize {
        for x in 0..w as usize {
            let mut acc = 0.0f32;
            for dy in 0..s {
                let row = (y * s + dy) * hw + x * s;
                for dx in 0..s {
                    acc += hi[row + dx];
                }
            }
            cov[y * w as usize + x] = acc * inv;
        }
    }
}

/// Static halo falloff for a w*w box. Tight and faint on purpose: a wide halo
/// sits under every dot as a fog, which is what makes a crisp matrix look
/// soft and low-resolution.
fn halo(w: i32) -> Vec<f32> {
    let mut g = vec![0.0f32; (w * w) as usize];
    let c = w as f32 * 0.5;
    for y in 0..w {
        for x in 0..w {
            let dx = x as f32 + 0.5 - c;
            let dy = y as f32 + 0.5 - c;
            let d = (dx * dx + dy * dy).sqrt() / (c * 0.62);
            g[(y * w + x) as usize] = (-d * d * 5.5).exp();
        }
    }
    g
}

// ------------------------------------------------------------------ orb
impl Orb {
    /// Work area of the monitor the orb is actually on, not the primary one.
    /// Recomputed every frame so a resolution change or a monitor switch is
    /// picked up without a restart.
    fn work_area(&self) -> RECT {
        unsafe {
            let hmon = MonitorFromWindow(hwnd_of(self.hwnd), MONITOR_DEFAULTTONEAREST);
            let mut mi = MONITORINFO {
                cbSize: std::mem::size_of::<MONITORINFO>() as u32,
                ..Default::default()
            };
            if GetMonitorInfoW(hmon, &mut mi).as_bool() && mi.rcWork.right > mi.rcWork.left {
                return mi.rcWork;
            }
            let mut wa: RECT = std::mem::zeroed();
            if SystemParametersInfoW(
                SPI_GETWORKAREA,
                0,
                Some(&mut wa as *mut RECT as *mut std::ffi::c_void),
                SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
            )
            .is_ok()
                && wa.right > wa.left
            {
                return wa;
            }
            RECT {
                left: 0,
                top: 0,
                right: GetSystemMetrics(SM_CXSCREEN),
                bottom: GetSystemMetrics(SM_CYSCREEN),
            }
        }
    }

    fn place(&self) -> (i32, i32) {
        let wa = self.work_area();
        (wa.right - self.w - MARGIN, wa.bottom - self.h - MARGIN)
    }

    /// Side length in *physical* pixels for a given DPI, never larger than the
    /// work area can hold.
    fn side_for(&self, dpi: u32) -> i32 {
        let wa = self.work_area();
        let want = (ORB_PX as f32 * dpi as f32 / 96.0).round() as i32;
        let cap = (wa.right - wa.left - MARGIN).min(wa.bottom - wa.top - MARGIN);
        want.clamp(96, cap.max(96))
    }

    /// (Re)allocate the DIB and the per-frame buffers at `side` physical pixels.
    /// Safe to call again on WM_DPICHANGED; the memory DC is kept.
    fn allocate(&mut self, side: i32) -> bool {
        unsafe {
            let screen = GetDC(None);
            let hdc = HDC(self.hdc as *mut std::ffi::c_void);
            let mut bmi = BITMAPINFO::default();
            bmi.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
            bmi.bmiHeader.biWidth = side;
            bmi.bmiHeader.biHeight = -side; // top-down
            bmi.bmiHeader.biPlanes = 1;
            bmi.bmiHeader.biBitCount = 32;
            bmi.bmiHeader.biCompression = BI_RGB.0;
            let mut bits: *mut std::ffi::c_void = std::ptr::null_mut();
            let bmp = match CreateDIBSection(hdc, &bmi, DIB_RGB_COLORS, &mut bits, None, 0) {
                Ok(b) => b,
                Err(_) => {
                    ReleaseDC(None, screen);
                    return false;
                }
            };
            let old = SelectObject(hdc, HGDIOBJ(bmp.0));
            ReleaseDC(None, screen);

            if self.bmp != 0 {
                SelectObject(hdc, HGDIOBJ(self.old_obj as *mut _));
                let _ = DeleteObject(HGDIOBJ(self.bmp as *mut _));
            }
            self.old_obj = old.0 as usize;
            self.bmp = bmp.0 as usize;
            self.bits = bits as usize;
            self.w = side;
            self.h = side;

            let px = (side * side) as usize;
            let hs = (side as f32 * SS).round() as usize;
            self.glow = halo(side);
            self.cov = vec![0.0f32; px];
            self.hi = vec![0.0f32; hs * hs];
            true
        }
    }

    /// Clicks in the transparent corners must not count as clicks on the orb.
    fn hit(&self, x: i32, y: i32) -> bool {
        let dx = x as f32 - self.w as f32 * 0.5;
        let dy = y as f32 - self.h as f32 * 0.5;
        let r = self.w as f32 * 0.4;
        dx * dx + dy * dy <= r * r
    }

    fn colour(&self) -> (f32, f32, f32) {
        colour_of(self.phase, self.ready_mix)
    }

    fn glow_strength(&self) -> f32 {
        match self.phase {
            Phase::Listening => 0.10 + 0.06 * self.amp,
            Phase::Thinking => 0.09,
            Phase::Ready => 0.06 + 0.12 * self.flash,
            Phase::Error => 0.08,
            Phase::Idle => 0.06,
        }
    }

    /// Envelope-follow the mic. `gate` floats with the room, so a noisy room
    /// does not leave the orb pinned at full scale.
    fn read_audio(&mut self) {
        let raw = f32::from_bits(RMS.load(Ordering::Relaxed));
        if !raw.is_finite() {
            return;
        }
        // Callbacks land every ~10 ms and frames every ~16 ms: hold the last
        // value for a beat rather than sampling a stair-stepped signal.
        self.rms = raw.max(self.rms * 0.72);
        if self.rms < self.floor {
            self.floor += (self.rms - self.floor) * 0.30;
        } else {
            self.floor += (self.rms - self.floor) * 0.0006;
        }
        self.floor = self.floor.clamp(0.0008, 0.05);
        self.gate = (self.floor * 3.0).max(0.0045);
        self.speaking = self.rms > self.gate;
        self.level = ((self.rms / (self.gate * 4.0)).clamp(0.0, 1.0)).powf(0.55);
    }

    fn amp_target(&self) -> f32 {
        match self.phase {
            Phase::Idle | Phase::Listening | Phase::Ready => {
                // Ready breathes exactly like Idle: the mic is closed and the
                // orb is back at rest, waiting out its life before dissolving.
                self.level.max(0.13 + 0.05 * (self.t * 0.9).sin())
            }
            Phase::Thinking => 0.42 + 0.28 * (0.5 + 0.5 * (self.t * 1.7).sin()),
            Phase::Error => 0.22,
        }
    }

    /// Advance one frame. Returns false once the orb has fully dissolved, so
    /// the caller can hide it *after* releasing the lock (see `wndproc`).
    fn tick(&mut self, now: Instant) -> bool {
        let dt = if self.reduce {
            // Reduced motion: no time-based motion at all, only state changes.
            0.0
        } else {
            (now - self.last).as_secs_f32().clamp(0.001, 0.05)
        };
        self.last = now;
        self.read_audio();

        if self.armed {
            self.phase = if self.speaking {
                Phase::Listening
            } else {
                Phase::Idle
            };
        }
        self.deadline = self
            .phase
            .life()
            .map(|s| self.deadline_start + Duration::from_secs_f32(s));

        self.flash = (self.flash - dt / 0.45).max(0.0);

        match self.deadline {
            Some(d) if d <= now => {
                self.opacity = (-(d - now).as_secs_f32() / FADE).clamp(0.0, 1.0);
                if self.opacity <= 0.001 {
                    return false;
                }
            }
            _ => self.opacity = 1.0,
        }

        let cur = self.phase.render();
        if self.reduce {
            // Snap the integrators, then skip frames that would look identical.
            self.weights = [0.0; RSTATES];
            self.weights[cur] = 1.0;
            self.amp = self.amp_target();
            self.scale = SCALE[cur];
            self.vel = 0.0;
            self.ready_mix += (crate::model_ready() as i32 as f32 - self.ready_mix) * 0.25;
            let sig = self.signature();
            if sig == self.sig && self.visible {
                return true;
            }
            self.sig = sig;
        } else {
            self.t += dt;
            let step = 1.0 - (1.0 - BLEND).powf(dt * 60.0);
            for (i, w) in self.weights.iter_mut().enumerate() {
                *w += ((i == cur) as i32 as f32 - *w) * step;
            }
            let target = self.amp_target();
            let rate = if target > self.amp { ATTACK } else { RELEASE };
            self.amp += (target - self.amp) * (1.0 - (1.0 - rate).powf(dt * 60.0));
            self.vel += (-STIFFNESS * (self.scale - SCALE[cur]) - DAMPING * self.vel) * dt;
            self.scale += self.vel * dt;
            // ~0.4 s crossfade from "loading" blue to "loaded" blue.
            self.ready_mix += (crate::model_ready() as i32 as f32 - self.ready_mix)
                * (1.0 - 0.05f32.powf(dt * 60.0));
        }

        unsafe {
            self.blit();
        }
        true
    }

    fn signature(&self) -> u64 {
        let q = |v: f32| (v.clamp(0.0, 1.0) * 8.0) as u64;
        ((self.phase as u64) << 32) | (q(self.ready_mix) << 16) | q(self.opacity.min(self.level))
    }

    /// Rasterise the matrix, composite the dots over the halo, push the DIB.
    unsafe fn blit(&mut self) {
        let (w, h) = (self.w, self.h);
        let look = Look {
            t: self.t,
            amp: self.amp,
            flash: self.flash,
            scale: self.scale,
            weights: self.weights,
        };
        let bits = self.bits as *mut u32;
        let (dr, dg, db) = self.colour();
        // The halo is the same hue pushed toward white: it reads as emitted
        // light, which is what keeps a mid-blue legible on a dark desktop.
        let (gr, gg, gb) = (dr * 0.45 + 0.55, dg * 0.45 + 0.55, db * 0.45 + 0.55);
        let gs = self.glow_strength();
        let o = self.opacity;
        let glow = &self.glow;

        rasterize(&mut self.cov, &mut self.hi, w, &look);
        let cov = &self.cov;

        let q = |v: f32| (((v * 255.0) + 0.5) as u32).min(255);
        let mut p = bits;
        for i in 0..glow.len() {
            let da = cov[i];
            let ga = glow[i] * gs;
            let inv = 1.0 - da;
            // Dots source-over the halo; both are premultiplied already.
            let a = (da + ga * inv) * o;
            unsafe {
                if a <= 0.002 {
                    *p = 0;
                } else {
                    // A 32-bit BI_RGB DIB holds [B, G, R, A] in memory, and a little-endian u32
                    // puts its least significant byte first - so blue goes unshifted and red goes
                    // at << 16. Getting these two the wrong way round renders the whole orb in
                    // the complementary hue, which is exactly what an "orange instead of blue"
                    // bug looks like.
                    *p = (q(a) << 24)
                        | (q((dr * da + gr * ga * inv) * o) << 16)
                        | (q((dg * da + gg * ga * inv) * o) << 8)
                        | q((db * da + gb * ga * inv) * o);
                }
                p = p.add(1);
            }
        }

        let (x, y) = self.place();
        let size = SIZE { cx: w, cy: h };
        let src = POINT { x: 0, y: 0 };
        let dst = POINT { x, y };
        let blend = BLENDFUNCTION {
            BlendOp: AC_SRC_OVER as u8,
            BlendFlags: 0,
            SourceConstantAlpha: 255,
            AlphaFormat: AC_SRC_ALPHA as u8,
        };
        unsafe {
            let screen = GetDC(None);
            let _ = UpdateLayeredWindow(
                hwnd_of(self.hwnd),
                screen,
                Some(&dst),
                Some(&size),
                HDC(self.hdc as *mut std::ffi::c_void),
                Some(&src),
                COLORREF(0),
                Some(&blend),
                ULW_ALPHA,
            );
            ReleaseDC(None, screen);
        }
        self.visible = true;
    }
}

// ------------------------------------------------------------------ window
unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    unsafe {
        match msg {
            WM_TIMER if wp.0 == TIMER_ID => {
                // Hooks run first and with no lock held: they call back into
                // this module, and a plain Mutex is not reentrant.
                if let Some(h) = hooks() {
                    (h.tick)();
                }
                let alive = {
                    let mut g = orb().lock().unwrap();
                    match g.as_mut() {
                        Some(o) => o.tick(Instant::now()),
                        None => false,
                    }
                };
                if !alive {
                    hide();
                }
                LRESULT(0)
            }
            WM_LBUTTONDOWN => {
                let (x, y) = lparam_xy(lp);
                // Two short critical sections, so the hook below runs unlocked.
                let clicked = {
                    let g = orb().lock().unwrap();
                    match g.as_ref() {
                        Some(o) if o.hit(x, y) => Some(o.phase),
                        _ => None,
                    }
                };
                if let (Some(p), Some(h)) = (clicked, hooks()) {
                    (h.click)(p);
                }
                LRESULT(0)
            }
            // Never pull focus away from the window being dictated into.
            WM_MOUSEACTIVATE => LRESULT(MA_NOACTIVATE as isize),
            // Dragged onto a monitor with a different scale: rebuild the DIB
            // and buffers at the new size, otherwise DWM upscales the old one
            // and the orb goes soft.
            WM_DPICHANGED => {
                let dpi = (wp.0 & 0xFFFF) as u32;
                let mut g = orb().lock().unwrap();
                if let Some(o) = g.as_mut() {
                    let side = o.side_for(dpi.max(96));
                    o.allocate(side);
                }
                LRESULT(0)
            }
            _ => DefWindowProcW(hwnd, msg, wp, lp),
        }
    }
}

/// Opt into per-monitor DPI awareness before any window exists.
///
/// Without this the process is virtualised: `GetDeviceCaps` reports a flat 96
/// no matter the real scale, so the DIB would be built too small and then
/// stretched by the compositor - a soft, low-resolution orb on any scaled
/// display. Failing here is survivable (the orb still shows, just unscaled),
/// so the caller only logs it.
pub fn dpi_aware() -> bool {
    // Ask for the most specific context first, then the older entry points.
    // Their return codes are not trustworthy: Windows answers ACCESS_DENIED
    // when the context is *already* set (which happens whenever the launching
    // shell pinned it), so success is confirmed by reading the context back
    // rather than by is_ok().
    unsafe {
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        let _ = SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
    }
    if awareness() >= DPI_AWARENESS_PER_MONITOR_AWARE.0 {
        return true;
    }
    unsafe {
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_SYSTEM_AWARE);
    }
    awareness() >= DPI_AWARENESS_SYSTEM_AWARE.0
}

/// This thread's current DPI awareness: 0 unaware, 1 system, 2 per-monitor.
fn awareness() -> i32 {
    unsafe { GetAwarenessFromDpiAwarenessContext(GetThreadDpiAwarenessContext()).0 }
}

/// What the orb actually runs at, for the startup log.
pub fn dpi_report() -> String {
    let a = match awareness() {
        0 => "unaware",
        1 => "system-aware",
        2 => "per-monitor-aware",
        _ => "per-monitor-aware-v2",
    };
    format!("dpi awareness: {a} ({})", awareness())
}

/// One line describing the live orb geometry, for the startup log.
pub fn geometry_report() -> String {
    let g = orb().lock().unwrap();
    match g.as_ref() {
        Some(o) => {
            let wa = o.work_area();
            format!(
                "orb: {}x{} device px @ dpi {}, work area {}x{}, corner ({},{})",
                o.w,
                o.h,
                unsafe { GetDpiForWindow(hwnd_of(o.hwnd)) },
                wa.right - wa.left,
                wa.bottom - wa.top,
                o.place().0,
                o.place().1,
            )
        }
        None => "orb: not created".to_string(),
    }
}

/// Create the window and its DIB. Idempotent.
pub fn ensure() -> bool {
    if orb().lock().unwrap().is_some() {
        return true;
    }
    unsafe {
        let hinst: HINSTANCE = GetModuleHandleW(None).unwrap().into();
        let cls = wide("DictationOrb");
        let wc = WNDCLASSW {
            lpfnWndProc: Some(wndproc),
            hInstance: hinst,
            lpszClassName: PCWSTR(cls.as_ptr()),
            ..Default::default()
        };
        if RegisterClassW(&wc) == 0 {
            return false;
        }

        // Provisional size; the DIB is built below from the real monitor DPI.
        let hwnd = match CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            PCWSTR(cls.as_ptr()),
            w!("Dictation"),
            WS_POPUP,
            0,
            0,
            ORB_PX,
            ORB_PX,
            None,
            None,
            hinst,
            None,
        ) {
            Ok(h) => h,
            Err(_) => return false,
        };

        let screen = GetDC(None);
        let hdc = CreateCompatibleDC(screen);
        ReleaseDC(None, screen);

        let now = Instant::now();
        let mut weights = [0.0f32; RSTATES];
        weights[R_IDLE] = 1.0;
        let mut o = Orb {
            hwnd: hwnd.0 as usize,
            hdc: hdc.0 as usize,
            old_obj: 0,
            bmp: 0,
            bits: 0,
            w: 0,
            h: 0,
            glow: Vec::new(),
            cov: Vec::new(),
            hi: Vec::new(),
            t: 0.0,
            amp: 0.0,
            scale: SCALE[R_IDLE],
            vel: 0.0,
            flash: 0.0,
            weights,
            ready_mix: 0.0,
            opacity: 1.0,
            phase: Phase::Idle,
            armed: false,
            visible: false,
            deadline_start: now,
            deadline: None,
            rms: 0.0,
            floor: 0.003,
            gate: 0.0045,
            speaking: false,
            level: 0.0,
            reduce: reduce_motion(),
            last: now,
            sig: u64::MAX,
        };
        // Real DPI of the monitor the window landed on, not the virtualised 96
        // a DPI-unaware process would be told.
        let side = o.side_for(GetDpiForWindow(hwnd).max(96));
        if !o.allocate(side) {
            return false;
        }
        *orb().lock().unwrap() = Some(o);
        true
    }
}

/// Restart the frame clock and (re)show the window.
fn reveal() {
    let hwnd = {
        let mut g = orb().lock().unwrap();
        let Some(o) = g.as_mut() else { return };
        let now = Instant::now();
        o.last = now;
        o.sig = u64::MAX;
        o.visible = false;
        o.deadline_start = now;
        o.opacity = 1.0;
        o.hwnd
    };
    unsafe {
        let _ = SetTimer(hwnd_of(hwnd), TIMER_ID, TIMER_MS, None);
        let _ = ShowWindow(hwnd_of(hwnd), SW_SHOWNOACTIVATE);
    }
}

fn set_phase(p: Phase) {
    reveal();
    if let Some(o) = orb().lock().unwrap().as_mut() {
        o.phase = p;
    }
}

// ------------------------------------------------------------------ public
/// Mic open. From here the orb picks Idle/Listening by itself from mic level.
pub fn set_armed(on: bool) {
    if on {
        reveal();
        if let Some(o) = orb().lock().unwrap().as_mut() {
            o.armed = true;
            o.phase = Phase::Idle;
            o.rms = 0.0;
            o.floor = 0.003;
            o.gate = 0.0045;
            o.speaking = false;
            o.level = 0.0;
            o.deadline = None;
        }
    } else if let Some(o) = orb().lock().unwrap().as_mut() {
        o.armed = false;
        o.rms = 0.0;
        o.level = 0.0;
        clear_level();
    }
}

pub fn thinking() {
    set_armed(false);
    set_phase(Phase::Thinking);
}

/// Text is final and already pasted. Clicking the orb copies it instead.
pub fn ready() {
    set_phase(Phase::Ready);
    if let Some(o) = orb().lock().unwrap().as_mut() {
        // Short pulse so the paste is acknowledged, then it settles back to the
        // resting idle look for the rest of its 5s life.
        o.flash = 0.6;
    }
}

pub fn fail() {
    set_phase(Phase::Error);
}

/// Click landed on a Ready orb: flash, and hand back the full life window.
pub fn copied() {
    if let Some(o) = orb().lock().unwrap().as_mut() {
        o.flash = 1.0;
        o.deadline_start = Instant::now();
        let hwnd = o.hwnd;
        unsafe {
            let _ = SetTimer(hwnd_of(hwnd), TIMER_ID, TIMER_MS, None);
        }
    }
}

pub fn hide() {
    let hwnd = {
        let mut g = orb().lock().unwrap();
        let Some(o) = g.as_mut() else { return };
        o.visible = false;
        o.armed = false;
        o.deadline = None;
        o.rms = 0.0;
        o.level = 0.0;
        clear_level();
        o.hwnd
    };
    unsafe {
        let _ = KillTimer(hwnd_of(hwnd), TIMER_ID);
        let _ = ShowWindow(hwnd_of(hwnd), SW_HIDE);
    }
}

/// Release GDI objects. Only useful at process exit.
pub fn destroy() {
    if let Some(o) = orb().lock().unwrap().take() {
        unsafe {
            SelectObject(
                HDC(o.hdc as *mut std::ffi::c_void),
                HGDIOBJ(o.old_obj as *mut _),
            );
            let _ = DeleteObject(HGDIOBJ(o.bmp as *mut _));
            let _ = DeleteDC(HDC(o.hdc as *mut std::ffi::c_void));
        }
    }
}
