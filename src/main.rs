use std::{
    collections::HashMap,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use clap::Parser;
use image::imageops::FilterType;
use ndarray::Array4;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor as OrtTensor,
};
use walkdir::WalkDir;

const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"];
// Used only if the model's ONNX metadata is missing `names` (ultralytics always embeds it).
const FALLBACK_CLASSES: &[&str] = &["manta", "non_fish", "other_fish"];
const FALLBACK_IMGSZ: u32 = 640;
const UNSURE_DIR: &str = "unsure";

/// Sort a folder of images into per-class sub-folders using a YOLO classification model (ONNX).
///
/// Class names, image size, channel count and tensor names are read from the model's
/// ONNX metadata, so the same tool works for any ultralytics classification model
/// (2-class, 3-class, ...) without recompiling.
#[derive(Parser, Debug)]
#[command(name = "localSort", version, about)]
struct Cli {
    /// Folder of images to sort. If omitted, a folder picker opens.
    input: Option<PathBuf>,

    /// Path to the ONNX model. If omitted, looks for `model.onnx` (then any single
    /// `*.onnx`) next to the images, in the working dir, and next to the executable.
    #[arg(long, value_name = "FILE")]
    model: Option<PathBuf>,

    /// Classify and report counts without moving or copying anything.
    #[arg(long)]
    dry_run: bool,

    /// Copy images into the class folders instead of moving them.
    #[arg(long)]
    copy: bool,

    /// Recurse into sub-directories (class/uncertain output folders are skipped).
    #[arg(long)]
    recursive: bool,

    /// Override the confidence threshold (0.0-1.0). Predictions below it go to the
    /// `unsure/` folder. Defaults to the value baked in at build time (0.6).
    #[arg(long, value_name = "FRACTION")]
    min_confidence: Option<f32>,

    /// Force grayscale preprocessing (only for models trained on grayscale images).
    #[arg(long)]
    grayscale: bool,
}

/// Everything we need to know about the loaded model, read from its ONNX metadata.
struct ModelInfo {
    class_names: Vec<String>,
    width: u32,
    height: u32,
    channels: usize,
    input_name: String,
}

/// Confidence threshold baked in at build time from the `MANTA_CONFIDENCE_THRESHOLD`
/// environment variable (set as a GitHub Actions variable on release builds). Falls back
/// to 0.6 for local/dev builds. The `--min-confidence` flag overrides it at runtime.
fn default_threshold() -> f32 {
    option_env!("MANTA_CONFIDENCE_THRESHOLD")
        .and_then(|s| s.trim().parse::<f32>().ok())
        .filter(|v| (0.0..=1.0).contains(v))
        .unwrap_or(0.6)
}

/// Resolve the image folder: use the CLI argument if given, otherwise open a GUI folder
/// picker. The picker (rfd) is compiled only on non-Linux targets (Windows/macOS), where
/// the app is typically launched by double-click; on Linux the folder must be passed as an
/// argument, which keeps GTK/Wayland system libraries out of the Linux build.
#[cfg(not(target_os = "linux"))]
fn resolve_input_dir(arg: Option<PathBuf>) -> Result<PathBuf> {
    match arg {
        Some(p) => Ok(p),
        None => rfd::FileDialog::new()
            .set_title("Select image directory")
            .pick_folder()
            .context("No directory selected"),
    }
}

#[cfg(target_os = "linux")]
fn resolve_input_dir(arg: Option<PathBuf>) -> Result<PathBuf> {
    arg.context(
        "No input folder given. On Linux, pass the image folder as an argument, e.g.:  localSort /path/to/images",
    )
}

fn main() {
    let cli = Cli::parse();
    // Launched without a folder argument (e.g. double-clicked) -> we open a GUI picker and
    // must keep the console open at the end so the summary is readable before it closes.
    let interactive = cli.input.is_none();

    let code = match run(cli) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("\nError: {e:#}");
            1
        }
    };

    if interactive {
        print!("\nPress Enter to exit...");
        let _ = io::stdout().flush();
        let mut buf = String::new();
        let _ = io::stdin().read_line(&mut buf);
    }
    std::process::exit(code);
}

fn run(cli: Cli) -> Result<()> {
    if let Some(thr) = cli.min_confidence {
        anyhow::ensure!(
            (0.0..=1.0).contains(&thr),
            "--min-confidence must be between 0.0 and 1.0 (got {thr})"
        );
    }
    // Build-time default, overridable at runtime with --min-confidence.
    let threshold = cli.min_confidence.unwrap_or_else(default_threshold);

    let input = resolve_input_dir(cli.input.clone())?;
    anyhow::ensure!(input.is_dir(), "Not a directory: {}", input.display());

    let model_path = match &cli.model {
        Some(p) => {
            anyhow::ensure!(p.is_file(), "Model not found: {}", p.display());
            p.clone()
        }
        None => discover_model(&input)?,
    };

    let mut session = load_session(&model_path)?;
    let model = read_model_info(&session);

    let mode = if cli.grayscale || model.channels == 1 {
        "grayscale"
    } else {
        "RGB"
    };
    println!("Model: {}", model_path.display());
    println!("  classes: [{}]", model.class_names.join(", "));
    println!(
        "  input: {}  size: {}x{}  channels: {}  mode: {}",
        model.input_name, model.width, model.height, model.channels, mode
    );
    println!("  confidence threshold: {threshold:.2}  (below -> {UNSURE_DIR}/)");

    // Destination dirs (one per class, plus uncertain). Computed even in --dry-run so that
    // already-sorted files are skipped and recursion never re-scans its own output.
    let mut dest_dirs: Vec<PathBuf> = model.class_names.iter().map(|c| input.join(c)).collect();
    dest_dirs.push(input.join(UNSURE_DIR));

    // Gather all candidate files up front so moves never affect the work list mid-run.
    let walker = if cli.recursive {
        WalkDir::new(&input)
    } else {
        WalkDir::new(&input).max_depth(1)
    };
    let mut files: Vec<PathBuf> = walker
        .min_depth(1)
        .into_iter()
        .filter_map(|e| e.ok())
        .map(|e| e.into_path())
        .filter(|p| p.is_file() && is_image(p))
        .filter(|p| !dest_dirs.iter().any(|d| p.starts_with(d)))
        .collect();
    files.sort();

    if files.is_empty() {
        println!("\nNo images found in {}", input.display());
        return Ok(());
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut errors = 0usize;

    for path in &files {
        let filename = path.file_name().unwrap().to_string_lossy().into_owned();

        let (idx, confidence) = match classify(&mut session, &model, path, cli.grayscale) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Classification error for {filename}: {e:#}");
                errors += 1;
                continue;
            }
        };

        let predicted = model
            .class_names
            .get(idx)
            .cloned()
            .unwrap_or_else(|| format!("class_{idx}"));
        let label = if confidence < threshold {
            UNSURE_DIR.to_string()
        } else {
            predicted.clone()
        };

        println!("{filename}  ->  {label}  ({:.0}%)", confidence * 100.0);

        if !cli.dry_run {
            let dest_dir = input.join(&label);
            if let Err(e) = fs::create_dir_all(&dest_dir) {
                eprintln!("Failed to create {}: {e}", dest_dir.display());
                errors += 1;
                continue;
            }
            let dest = unique_path(&dest_dir, &filename);
            if let Err(e) = place_file(path, &dest, cli.copy) {
                eprintln!("Failed to place {filename}: {e}");
                errors += 1;
                continue;
            }
        }

        *counts.entry(label).or_insert(0) += 1;
    }

    print_summary(&model, &counts, errors, &cli);
    Ok(())
}

fn print_summary(
    model: &ModelInfo,
    counts: &HashMap<String, usize>,
    errors: usize,
    cli: &Cli,
) {
    let header = if cli.dry_run {
        "DRY RUN - no files changed".to_string()
    } else if cli.copy {
        "Summary (copied)".to_string()
    } else {
        "Summary (moved)".to_string()
    };
    println!("\n--- {header} ---");
    for class in &model.class_names {
        println!("  {class}: {}", counts.get(class).copied().unwrap_or(0));
    }
    println!("  {UNSURE_DIR}: {}", counts.get(UNSURE_DIR).copied().unwrap_or(0));
    if errors > 0 {
        println!("  errors: {errors}");
    }
}

fn load_session(model_path: &Path) -> Result<Session> {
    Session::builder()
        .map_err(|e| anyhow::anyhow!("SessionBuilder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("OptLevel: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load model {}: {e}", model_path.display()))
}

/// Reads class names, image size, channel count and the input tensor name from the
/// ONNX metadata that ultralytics embeds on export. Falls back to sensible defaults.
fn read_model_info(session: &Session) -> ModelInfo {
    let meta = session.metadata().ok();
    let custom = |key: &str| meta.as_ref().and_then(|m| m.custom(key));

    let class_names = custom("names")
        .and_then(|s| parse_names(&s))
        .unwrap_or_else(|| FALLBACK_CLASSES.iter().map(|s| s.to_string()).collect());

    let (height, width) = custom("imgsz")
        .and_then(|s| parse_imgsz(&s))
        .unwrap_or((FALLBACK_IMGSZ, FALLBACK_IMGSZ));

    let channels = custom("channels")
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|c| *c > 0)
        .unwrap_or(3);

    let input_name = session
        .inputs()
        .first()
        .map(|o| o.name().to_string())
        .unwrap_or_else(|| "images".to_string());

    ModelInfo { class_names, width, height, channels, input_name }
}

/// Parse ultralytics' `names` metadata, e.g. `{0: 'manta', 1: 'non_manta'}`, into a
/// Vec ordered by class index. Tolerant of single or double quotes and extra whitespace.
fn parse_names(raw: &str) -> Option<Vec<String>> {
    let bytes = raw.as_bytes();
    let mut pairs: Vec<(usize, String)> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let idx: usize = raw[start..i].parse().ok()?;
        while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b':') {
            i += 1;
        }
        if i < bytes.len() && (bytes[i] == b'\'' || bytes[i] == b'"') {
            let quote = bytes[i];
            i += 1;
            let lstart = i;
            while i < bytes.len() && bytes[i] != quote {
                i += 1;
            }
            pairs.push((idx, raw[lstart..i].to_string()));
            i += 1; // consume closing quote
        }
    }
    if pairs.is_empty() {
        return None;
    }
    pairs.sort_by_key(|(idx, _)| *idx);
    Some(pairs.into_iter().map(|(_, label)| label).collect())
}

/// Parse `imgsz` metadata (e.g. `[640, 640]` or `640`) into (height, width).
fn parse_imgsz(raw: &str) -> Option<(u32, u32)> {
    let nums: Vec<u32> = raw
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect();
    match nums.as_slice() {
        [h, w, ..] => Some((*h, *w)),
        [s] => Some((*s, *s)),
        _ => None,
    }
}

fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Find the model when `--model` is not given: prefer `model.onnx`, otherwise the single
/// `*.onnx` in a candidate dir. Errors clearly when none or ambiguously many are present.
fn discover_model(input: &Path) -> Result<PathBuf> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    let cwd = std::env::current_dir().ok();

    let mut dirs: Vec<PathBuf> = Vec::new();
    for d in [Some(input.to_path_buf()), cwd, exe_dir].into_iter().flatten() {
        if d.is_dir() && !dirs.iter().any(|e| e == &d) {
            dirs.push(d);
        }
    }

    // 1. Exact `model.onnx`.
    for d in &dirs {
        let p = d.join("model.onnx");
        if p.is_file() {
            return Ok(p);
        }
    }

    // 2. A single `*.onnx` in a candidate dir. Remember ambiguity for a better error.
    let mut ambiguous: Option<(PathBuf, Vec<String>)> = None;
    for d in &dirs {
        let onnx: Vec<PathBuf> = fs::read_dir(d)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.is_file()
                    && p.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("onnx"))
                        == Some(true)
            })
            .collect();
        match onnx.len() {
            1 => return Ok(onnx.into_iter().next().unwrap()),
            n if n > 1 && ambiguous.is_none() => {
                let names = onnx
                    .iter()
                    .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                    .collect();
                ambiguous = Some((d.clone(), names));
            }
            _ => {}
        }
    }

    if let Some((dir, names)) = ambiguous {
        bail!(
            "Multiple .onnx models in {}:\n  {}\nPick one with --model <file>.",
            dir.display(),
            names.join("\n  ")
        );
    }

    bail!(
        "No ONNX model found in:\n  {}\nExport one with:  yolo export model=model.pt format=onnx\n\
         then place `model.onnx` next to the images or the executable (or pass --model <file>).",
        dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join("\n  ")
    )
}

/// Build a non-colliding destination path by appending ` (1)`, ` (2)`, ... before the extension.
fn unique_path(dir: &Path, filename: &str) -> PathBuf {
    let candidate = dir.join(filename);
    if !candidate.exists() {
        return candidate;
    }
    let p = Path::new(filename);
    let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let ext = p
        .extension()
        .map(|e| format!(".{}", e.to_string_lossy()))
        .unwrap_or_default();
    let mut n = 1;
    loop {
        let candidate = dir.join(format!("{stem} ({n}){ext}"));
        if !candidate.exists() {
            return candidate;
        }
        n += 1;
    }
}

fn place_file(src: &Path, dst: &Path, copy: bool) -> std::io::Result<()> {
    if copy {
        fs::copy(src, dst).map(|_| ())
    } else {
        // rename is atomic within a filesystem; fall back to copy+remove across mounts.
        match fs::rename(src, dst) {
            Ok(()) => Ok(()),
            Err(_) => {
                fs::copy(src, dst)?;
                fs::remove_file(src)
            }
        }
    }
}

/// Resize shorter edge to `target` (preserving aspect ratio), then center-crop `target`x`target`.
/// Returns (resized_w, resized_h, crop_x0, crop_y0). Mirrors torchvision `Resize(s)+CenterCrop(s)`,
/// which is exactly what ultralytics classification inference does.
fn resize_crop_dims(src_w: u32, src_h: u32, target: u32) -> (u32, u32, u32, u32) {
    let (new_w, new_h) = if src_w <= src_h {
        (target, ((src_h as f64) * (target as f64) / (src_w as f64)).round() as u32)
    } else {
        (((src_w as f64) * (target as f64) / (src_h as f64)).round() as u32, target)
    };
    let x0 = new_w.saturating_sub(target) / 2;
    let y0 = new_h.saturating_sub(target) / 2;
    (new_w.max(target), new_h.max(target), x0, y0)
}

/// Returns (class_index, confidence). Preprocessing matches ultralytics classification exactly:
/// RGB (or grayscale if forced), resize shorter edge to imgsz + center-crop, scale to [0,1], NCHW.
fn classify(
    session: &mut Session,
    model: &ModelInfo,
    path: &Path,
    force_gray: bool,
) -> Result<(usize, f32)> {
    let img = image::open(path).with_context(|| format!("Failed to open {}", path.display()))?;
    let c = model.channels;
    let target = model.height.min(model.width); // classification models use a square crop

    let mut input = Array4::<f32>::zeros((1, c, target as usize, target as usize));
    if force_gray || c == 1 {
        let buf = img.to_luma8();
        let (nw, nh, x0, y0) = resize_crop_dims(buf.width(), buf.height(), target);
        let resized = image::imageops::resize(&buf, nw, nh, FilterType::Triangle);
        for ty in 0..target {
            for tx in 0..target {
                let v = resized.get_pixel(x0 + tx, y0 + ty)[0] as f32 / 255.0;
                for ch in 0..c {
                    input[[0, ch, ty as usize, tx as usize]] = v;
                }
            }
        }
    } else {
        let buf = img.to_rgb8();
        let (nw, nh, x0, y0) = resize_crop_dims(buf.width(), buf.height(), target);
        let resized = image::imageops::resize(&buf, nw, nh, FilterType::Triangle);
        for ty in 0..target {
            for tx in 0..target {
                let pixel = resized.get_pixel(x0 + tx, y0 + ty);
                for ch in 0..c.min(3) {
                    input[[0, ch, ty as usize, tx as usize]] = pixel[ch] as f32 / 255.0;
                }
            }
        }
    }

    let tensor = OrtTensor::from_array(input)
        .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {e}"))?;
    let outputs = session
        .run(ort::inputs![model.input_name.as_str() => tensor])
        .map_err(|e| anyhow::anyhow!("Model inference failed: {e}"))?;
    let (_shape, scores) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to extract output tensor: {e}"))?;

    let probs = to_probabilities(scores);
    let (idx, conf) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, p)| (i, *p))
        .unwrap_or((0, 0.0));
    Ok((idx, conf))
}

/// ultralytics classification heads already apply softmax in-graph, so the ONNX output is
/// usually a probability distribution. Use it directly when it is one; otherwise apply a
/// numerically-stable softmax (so logit-output models still get a calibrated confidence).
fn to_probabilities(scores: &[f32]) -> Vec<f32> {
    let sum: f32 = scores.iter().sum();
    let looks_like_probs =
        scores.iter().all(|&v| (-1e-6..=1.0 + 1e-6).contains(&v)) && (sum - 1.0).abs() < 1e-3;
    if looks_like_probs {
        return scores.to_vec();
    }
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&v| (v - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / total).collect()
}
