# ml-runner

Sorts a folder of images into per-class subfolders using a YOLO ONNX classifier. Class names
come from the model's metadata, so it works with any ultralytics classification model.
Low-confidence images go to an `unsure` folder.

## Quick start

Download the binary for your OS from the
[latest release](https://github.com/theSearchLife/RunML/releases/latest), then run it from
that folder — pass the model's repo and the images folder:

```bat
:: Windows (cmd)
ml-runner.exe --model=theSearchLife/MantaWatch "C:\path\to\photos"
```

PowerShell: `.\ml-runner.exe …` &nbsp;·&nbsp; macOS/Linux: `./ml-runner … /path/to/photos`

The model is downloaded from that repo's latest release and cached; images are sorted into
per-class folders plus `unsure`.

- [Usage guide](docs/usage.md)
- [Technical docs](docs/architecture.md)

## Highlights

- **Model-agnostic** — classes and input size are read from the ONNX metadata.
- Model fetched & cached from a GitHub release repo (`--model=Org/Repo`); the binary
  **self-updates** from its own releases.
- **Self-contained** (ONNX Runtime statically linked); Linux, Windows, macOS (Apple Silicon).

