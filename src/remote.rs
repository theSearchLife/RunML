//! GitHub release integration:
//!   * download & cache the ONNX model from a model repo (`--model=Org/Repo`), and
//!   * self-update the `ml-runner` binary from its own repo.
//!
//! All HTTP uses ureq + rustls, so no platform needs OpenSSL or other system libraries.

use std::{
    fs,
    io::Read,
    path::PathBuf,
};

use anyhow::{Context, Result};
use serde::Deserialize;

const USER_AGENT: &str = concat!("ml-runner/", env!("CARGO_PKG_VERSION"));

#[derive(Debug, Deserialize)]
pub struct Release {
    pub tag_name: String,
    #[serde(default)]
    pub assets: Vec<Asset>,
}

#[derive(Debug, Deserialize)]
pub struct Asset {
    pub name: String,
    pub browser_download_url: String,
}

/// A GitHub token from the environment (lifts the 60 req/h anonymous rate limit and allows
/// private repos). Optional — everything works against public repos without it.
fn token() -> Option<String> {
    ["GITHUB_TOKEN", "GH_TOKEN"]
        .into_iter()
        .find_map(|k| std::env::var(k).ok())
        .filter(|t| !t.is_empty())
}

fn get(url: &str) -> ureq::Request {
    let mut req = ureq::get(url).set("User-Agent", USER_AGENT);
    if let Some(tok) = token() {
        req = req.set("Authorization", &format!("Bearer {tok}"));
    }
    req
}

fn api_json<T: serde::de::DeserializeOwned>(url: &str) -> Result<T> {
    get(url)
        .set("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| anyhow::anyhow!("request to {url} failed: {e}"))?
        .into_json::<T>()
        .with_context(|| format!("parsing JSON response from {url}"))
}

fn download_bytes(url: &str) -> Result<Vec<u8>> {
    let resp = get(url)
        .call()
        .map_err(|e| anyhow::anyhow!("download from {url} failed: {e}"))?;
    let mut buf = Vec::new();
    resp.into_reader()
        .read_to_end(&mut buf)
        .context("reading response body")?;
    Ok(buf)
}

/// Latest published release of `owner/repo` (skips drafts/prereleases, per the GitHub API).
pub fn latest_release(repo: &str) -> Result<Release> {
    api_json(&format!("https://api.github.com/repos/{repo}/releases/latest"))
}

/// Per-user cache root, e.g. `%LOCALAPPDATA%\ml-runner` (Windows) or `~/.cache/ml-runner`.
fn cache_root() -> Result<PathBuf> {
    #[cfg(target_os = "windows")]
    let base = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(not(target_os = "windows"))]
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")));
    Ok(base
        .context("could not determine a cache directory")?
        .join("ml-runner"))
}

/// Ensure the model from `repo`'s latest release is available locally, refreshing the cache
/// when the repo has a newer release. Returns the path to the cached `model.onnx`. When the
/// network is unavailable, falls back to the cached copy if one exists.
pub fn ensure_model(repo: &str) -> Result<PathBuf> {
    let dir = cache_root()?.join("models").join(repo.replace('/', "__"));
    let model_path = dir.join("model.onnx");
    let tag_path = dir.join("release-tag.txt");

    match latest_release(repo) {
        Ok(rel) => {
            let cached_tag = fs::read_to_string(&tag_path).ok();
            if model_path.is_file() && cached_tag.as_deref() == Some(rel.tag_name.as_str()) {
                println!("Model: cached {} ({})", repo, rel.tag_name);
                return Ok(model_path);
            }
            let asset = rel
                .assets
                .iter()
                .find(|a| a.name.eq_ignore_ascii_case("model.onnx"))
                .or_else(|| rel.assets.iter().find(|a| a.name.to_lowercase().ends_with(".onnx")))
                .with_context(|| {
                    format!("release {} of {repo} has no .onnx asset", rel.tag_name)
                })?;
            println!("Downloading model {} from {repo} ({}) ...", asset.name, rel.tag_name);
            let bytes = download_bytes(&asset.browser_download_url)?;
            fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
            fs::write(&model_path, &bytes)
                .with_context(|| format!("writing {}", model_path.display()))?;
            let _ = fs::write(&tag_path, &rel.tag_name);
            println!("Model cached at {}", model_path.display());
            Ok(model_path)
        }
        Err(e) => {
            if model_path.is_file() {
                eprintln!("Note: couldn't check {repo} for a newer model ({e}); using cached copy.");
                Ok(model_path)
            } else {
                Err(e.context(format!("no cached model for {repo} and the download failed")))
            }
        }
    }
}

/// Download a release archive and extract the single binary it holds into a temp file.
/// Windows assets are `.zip`; Linux assets are `.tar.gz`. (Not used on macOS, which ships a
/// `.pkg` installer and doesn't self-replace.)
#[cfg(not(target_os = "macos"))]
fn download_and_extract(url: &str, is_zip: bool) -> Result<PathBuf> {
    use std::io::{copy, Cursor};
    let bytes = download_bytes(url)?;
    let dest = std::env::temp_dir().join(format!("ml-runner-update-{}", std::process::id()));
    let mut out =
        fs::File::create(&dest).with_context(|| format!("creating {}", dest.display()))?;
    if is_zip {
        let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).context("opening downloaded .zip")?;
        let mut entry = zip.by_index(0).context("the .zip is empty")?;
        copy(&mut entry, &mut out).context("extracting the binary from the .zip")?;
    } else {
        let decoder = flate2::read::GzDecoder::new(Cursor::new(bytes));
        let mut archive = tar::Archive::new(decoder);
        let mut entry = archive
            .entries()
            .context("reading the .tar.gz")?
            .next()
            .context("the .tar.gz is empty")?
            .context("reading the .tar.gz entry")?;
        copy(&mut entry, &mut out).context("extracting the binary from the .tar.gz")?;
    }
    drop(out);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&dest, fs::Permissions::from_mode(0o755));
    }
    Ok(dest)
}

/// Parse a `vMAJOR.MINOR.PATCH` (or `MAJOR.MINOR`) tag into a comparable tuple.
fn parse_version(s: &str) -> Option<(u32, u32, u32)> {
    let mut parts = s.trim().trim_start_matches('v').split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    Some((major, minor, patch))
}

/// Check `repo` for a release newer than the running version and, if found, replace the
/// running executable in place. Returns `Some(new_version)` when an update was applied.
/// Errors are non-fatal to the caller (treat as "update skipped").
pub fn self_update(repo: &str) -> Result<Option<String>> {
    let current =
        parse_version(env!("CARGO_PKG_VERSION")).context("unparseable current version")?;
    let rel = latest_release(repo)?;
    let latest = parse_version(&rel.tag_name)
        .with_context(|| format!("unparseable release tag `{}`", rel.tag_name))?;
    if latest <= current {
        return Ok(None);
    }

    // macOS ships a `.pkg` installer, which can't be swapped in place — just notify.
    #[cfg(target_os = "macos")]
    {
        println!(
            "A newer ml-runner ({}) is available — download it from https://github.com/{repo}/releases/latest",
            rel.tag_name
        );
        Ok(None)
    }
    // Windows/Linux: download the release archive, extract the binary, replace ourselves.
    #[cfg(not(target_os = "macos"))]
    {
        let (suffix, is_zip) = match std::env::consts::OS {
            "windows" => ("windows_amd64.zip", true),
            _ => ("linux_amd64.tar.gz", false),
        };
        let asset = rel
            .assets
            .iter()
            .find(|a| a.name.ends_with(suffix))
            .with_context(|| format!("release {} has no `*{suffix}` asset", rel.tag_name))?;
        let tmp = download_and_extract(&asset.browser_download_url, is_zip)?;
        self_replace::self_replace(&tmp).context("replacing the running executable")?;
        let _ = fs::remove_file(&tmp);
        Ok(Some(rel.tag_name))
    }
}
