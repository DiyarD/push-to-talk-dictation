//! Embeds `manifest.xml` so the process starts out per-monitor-DPI-aware.
//!
//! This has to happen at link time. Without an explicit manifest the MSVC
//! linker synthesises one that pins DPI awareness to "system aware", after
//! which `SetProcessDpiAwarenessContext` fails with ACCESS_DENIED and the orb's
//! DIB gets sized for the wrong scale and stretched by the compositor.

fn main() {
    println!("cargo:rerun-if-changed=manifest.xml");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-link-arg-bins=/MANIFEST:EMBED");
    println!("cargo:rustc-link-arg-bins=/MANIFESTINPUT:manifest.xml");
}
