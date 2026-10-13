//! Compile Berkeley SoftFloat 3 (third_party/berkeley-softfloat-3) with its RISC-V
//! specialization, the configuration Spike uses, plus a small shim for its thread-local
//! state. The source list comes from upstream's FAST_INT64 Makefile template.

use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../../third_party/berkeley-softfloat-3");
    let source = root.join("source");
    let makefile = root.join("build/template-FAST_INT64/Makefile");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    if !source.exists() {
        panic!("{} is missing; run `git submodule update --init`", source.display());
    }

    std::fs::write(
        out.join("platform.h"),
        "#define LITTLEENDIAN 1\n\
         #define INLINE static inline\n\
         #define SOFTFLOAT_BUILTIN_CLZ 1\n\
         #define SOFTFLOAT_INTRINSIC_INT128 1\n\
         #include \"opts-GCC.h\"\n",
    )
    .unwrap();

    // Object lists: lines like "  s_add128$(OBJ) \" under OBJS_PRIMITIVES, OBJS_SPECIALIZE,
    // and OBJS_OTHERS. Specialization files live in source/RISCV.
    let text = std::fs::read_to_string(&makefile).unwrap();
    let mut files = Vec::new();
    let mut section = None;
    for line in text.lines() {
        if let Some(name) = line.split(" =").next().filter(|_| line.starts_with("OBJS_")) {
            section = Some(name.to_string());
            continue;
        }
        let Some(sec) = &section else { continue };
        let Some(obj) =
            line.trim().strip_suffix("$(OBJ) \\").or(line.trim().strip_suffix("$(OBJ)"))
        else {
            if line.trim().is_empty() {
                section = None;
            }
            continue;
        };
        let dir = if sec == "OBJS_SPECIALIZE" { source.join("RISCV") } else { source.clone() };
        files.push(dir.join(format!("{obj}.c")));
    }
    assert!(files.len() > 100, "parsed only {} sources from {}", files.len(), makefile.display());

    let mut build = cc::Build::new();
    // Match Rust's deployment target so the linker does not warn about newer objects.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        let min = std::env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "11.0".into());
        build.flag(format!("-mmacosx-version-min={min}"));
    }
    build
        .files(&files)
        .file("src/shim.c")
        .include(&out)
        .include(source.join("RISCV"))
        .include(source.join("include"))
        .define("SOFTFLOAT_FAST_INT64", None)
        .define("SOFTFLOAT_ROUND_ODD", None)
        .define("INLINE_LEVEL", "5")
        .define("SOFTFLOAT_FAST_DIV32TO16", None)
        .define("SOFTFLOAT_FAST_DIV64TO32", None)
        .define("THREAD_LOCAL", "_Thread_local")
        .flag_if_supported("-Wno-unused-parameter")
        .warnings(false)
        .compile("softfloat");

    println!("cargo:rerun-if-changed=src/shim.c");
    println!("cargo:rerun-if-changed={}", makefile.display());
}
