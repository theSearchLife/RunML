# ml-runner — Technical Documentation

Internal reference for developers/maintainers. For end-user instructions see
[usage.md](usage.md).

## Overview

`ml-runner` is a single-binary Rust CLI that runs a YOLO image-classification ONNX model over
a folder of images and moves each image into a subfolder named after the predicted class
(plus an `unsure` folder for low-confidence predictions). ONNX Runtime is **statically
linked**, so each release is one self-contained executable — no sidecar libraries.

## Repository layout

| Path | Purpose |
|---|---|
| `src/main.rs` | CLI (clap), the run pipeline, image preprocessing, ONNX inference, file sorting. |
| `src/remote.rs` | GitHub release integration: download & cache the model, and self-update the binary. |
| `Cargo.toml` | Package metadata and dependencies. |
| `.github/workflows/release.yml` | Release CI: build → (macOS) sign & notarize → publish. |
| `.github/actions/sign-notarize-macos/` | Composite action: macOS codesign + notarization. |
| `docs/` | `usage.md` (end users) and this file. |

## Run pipeline (`src/main.rs`)

1. Parse arguments (`clap`).
2. **Self-update check** — `remote::self_update(TOOL_REPO)` unless `--no-update` (non-fatal).
3. **Threshold** — `--min-confidence`, else `default_threshold()`.
4. **Input dir** — the positional `IMAGES_DIR`, else the current directory.
5. **Model** — `resolve_model`: a repo slug → download & cache; a local `.onnx` path → use it;
   omitted → `discover_model` searches next to the images / cwd / executable.
6. **Load** — open the ONNX session (`ort`) and read metadata (`read_model_info`).
7. **Sort** — walk the folder (`walkdir`, skipping the output subfolders); for each image:
   preprocess → run → argmax → route to the class folder, or `unsure` if the top
   probability is below the threshold → move (or `--copy`, or nothing on `--dry-run`).
8. Print a per-folder count summary.

`main()` wraps `run()` so it can print a friendly error and (unless `--no-pause`) wait for
Enter before exiting — handy when the exe is launched from a file manager.

## Preprocessing & inference

The preprocessing **exactly mirrors ultralytics classification inference** — this was
validated to reproduce the source `.pt` model's predictions (~94% on a 600-image labelled
set):

- **RGB**, resize the **shorter edge** to `imgsz`, then **center-crop** `imgsz × imgsz`.
- **Antialiased** resize (the `image` crate's `Triangle` filter). This matters: without
  antialiasing, high-frequency photos diverge from the reference.
- Scale to `[0, 1]` (just `/255`, no ImageNet mean/std), laid out as an NCHW `f32` tensor.
- `--grayscale` forces single-channel luma (only for grayscale-trained models). The manta
  model is RGB despite legacy "grayscale" filenames.

**Model metadata** (`read_model_info`) is read from what ultralytics embeds on export:
`names` (class list), `imgsz`, `channels`, and the input tensor name — with sensible
fallbacks. This is what makes the tool model-agnostic: **output folders mirror the model's
class names**. The classify head applies softmax in-graph, so outputs are used directly as
probabilities; a numerically-stable softmax is applied only if the output isn't already a
distribution (`to_probabilities`).

> The exported ONNX has a static `batch=1`, so images are processed one at a time.

## Model distribution (`src/remote.rs`)

`--model=Org/Repo` resolves the model from a GitHub repository's releases:

- `GET /repos/{repo}/releases/latest`, find the asset named `model.onnx` (else the first
  `*.onnx`), download it, and cache under
  `<cache>/ml-runner/models/<org>__<repo>/model.onnx` alongside a `release-tag.txt`.
  (`<cache>` = `%LOCALAPPDATA%` on Windows, `$XDG_CACHE_HOME` or `~/.cache` elsewhere.)
- Each run compares the cached tag to the latest release; it re-downloads only when the tag
  changed. Offline (or on error) it falls back to the cached copy.

## Self-update (`src/remote.rs`)

On startup the tool checks its **own** repo (`const TOOL_REPO = "theSearchLife/RunML"`):
`parse_version` compares the latest release tag against `env!("CARGO_PKG_VERSION")`. If newer:

- **Windows/Linux** — download the release archive (`ml-runner_windows_amd64.zip` /
  `ml-runner_linux_amd64.tar.gz`), extract the binary (`zip` / `flate2`+`tar`), and replace
  the running executable via the `self-replace` crate.
- **macOS** — the release is a `.pkg` installer, which can't be swapped in place, so it just
  prints a notice pointing at the releases page (no silent self-update on macOS).

All errors are non-fatal (treated as "update skipped").

## Confidence threshold

The default threshold is **baked at build time** from the `MANTA_CONFIDENCE_THRESHOLD`
environment variable (`option_env!` in `default_threshold()`, default `0.6`); release CI sets
it from a repository **Variable**. `--min-confidence` overrides it at runtime. Predictions
below the effective threshold go to `unsure/`.

## HTTP / auth

All GitHub access uses `ureq` with **rustls** (no OpenSSL). A `GITHUB_TOKEN` or `GH_TOKEN`
environment variable, if present, is sent as a bearer token — needed for private repos or to
lift the 60 req/hour anonymous rate limit.

## Dependencies

| Crate | Role |
|---|---|
| `anyhow`, `clap` (derive) | error handling, CLI |
| `image`, `ndarray` | decode/resize, tensor layout |
| `ort` (`download-binaries`, `copy-dylibs`) | ONNX Runtime — statically linked |
| `ureq` (rustls), `serde` | GitHub HTTP + JSON |
| `self-replace` | in-place binary update (Windows/Linux) |
| `zip`, `tar`, `flate2` | extract the self-update archive (target-gated to non-macOS) |
| `walkdir` | directory traversal |

Everything is pure Rust + rustls, so no platform needs system libraries (no OpenSSL, no GTK).

## Build

```bash
cargo build --release           # → target/release/ml-runner(.exe)
MANTA_CONFIDENCE_THRESHOLD=0.7 cargo build --release   # bake a custom default threshold
```

## Release CI (`.github/workflows/release.yml`)

- **Trigger:** push to `main` touching `Cargo.toml` or the workflow; or `workflow_dispatch`.
- **`get-version`** reads `version` from `Cargo.toml` → tag `v<version>`.
- **`build`** (matrix) → `x86_64-unknown-linux-gnu` (ubuntu), `x86_64-pc-windows-msvc`
  (windows), `aarch64-apple-darwin` (macos-14). Each job packages its asset and uploads it;
  the macOS leg is `continue-on-error` (30-min timeout) so a signing hiccup can't block the
  Linux/Windows release.
- **Packaging** — Linux → `ml-runner_linux_amd64.tar.gz`, Windows →
  `ml-runner_windows_amd64.zip` (each holds the bare `ml-runner`/`ml-runner.exe`); macOS →
  `ml-runner_macos_arm64.pkg` (a signed, notarized installer). Asset names carry no version —
  the version lives in the release tag.
- **`release`** downloads all artifacts and publishes a GitHub Release
  (`softprops/action-gh-release`).

### macOS packaging & signing

The inline macOS steps import two certs into a temporary keychain, `codesign` the binary with
**Developer ID Application** (hardened runtime + timestamp), build a `.pkg` with `pkgbuild`
(install location `/usr/local/bin`) signed with **Developer ID Installer**, then notarize the
`.pkg` (`xcrun notarytool --wait`) and `stapler staple` it (a `.pkg` can be stapled, so
Gatekeeper verifies offline). Required repository **Secrets** (all 10 from the bundle):
`MACOS_CERTIFICATE` (+`_PWD`), `MACOS_SIGNING_IDENTITY`, `MACOS_INSTALLER_CERTIFICATE`
(+`_PWD`), `MACOS_INSTALLER_SIGNING_IDENTITY`, `KEYCHAIN_PASSWORD`, `APPLE_API_KEY` (+`_ID`,
+`_ISSUER_ID`). The API key secret may be raw PEM or base64 (handled).

### No Intel macOS

`ort` ships **no `x86_64-apple-darwin` ONNX Runtime prebuilt**, so an Intel macOS binary
can't be built with the current backend (confirmed: a native build on an Intel runner fails
with `no prebuilt binaries for the target x86_64-apple-darwin`, while arm64 builds fine).
GitHub's `macos-13` Intel runners were also retired (2025-12-04). Adding Intel later would
require a different ONNX backend (e.g. the pure-Rust `ort-tract`) or building ONNX Runtime
from source.
