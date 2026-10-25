# BFF Camera Demo v0.1 — Third-Party Licence Inventory

This file is the canonical inventory of every third-party licence shipped
in the BFF Camera Demo v0.1 deliverables (Windows installer, Docker
images, and source tree). The application source code authored in this
repository is **Apache-2.0** unless a sub-directory declares otherwise.

If you add, remove, or replace any dependency, update the table below
and re-run the licence audit. This file is read by:

- `scripts/build_installer.py` (bundles the text with the NSIS installer
  EULA page),
- `scripts/build_pyinstaller.spec` (PyInstaller includes it as
  `LICENCE.md` in the dist output),
- `frontend/components/about.py` (renders the version + licence summary
  in the About tab).

## 1. Application Source

| Component | Licence | Copyright |
|---|---|---|
| `backend/` | Apache-2.0 | BFF Demo Team |
| `frontend/` | Apache-2.0 | BFF Demo Team |
| `ml/` | Apache-2.0 | BFF Demo Team |
| `scripts/` | Apache-2.0 | BFF Demo Team |
| `system-map-arsitektur.MD`, `backend.MD`, `frontend.MD`, `root README.md` | CC BY 4.0 | BFF Demo Team |

## 2. Python Runtime Dependencies (pinned via `pyproject.toml`)

| Package | Licence | Notes |
|---|---|---|
| `fastapi` | MIT | |
| `uvicorn` | BSD-3 | |
| `pydantic` | MIT | |
| `pydantic-settings` | MIT | |
| `python-dotenv` | BSD-3 | |
| `httpx` | BSD-3 | |
| `loguru` | MIT | |
| `python-keyring` | MIT | OS credential manager backend |
| `pynacl` | Apache-2.0 | libsodium binding (secretbox) |
| `boto3`, `botocore` | Apache-2.0 | AWS S3 client, used against Supabase S3 |
| `supabase` | MIT | |
| `sqlmodel` | MIT | |
| `aiosqlite` | MIT | |
| `pyyaml` | MIT | |
| `pillow` | HPND | |
| `numpy` | BSD-3 | |
| `opencv-python-headless` | Apache-2.0 | |
| `onnxruntime` | MIT | CPU build (base) |
| `onnxruntime-directml` | MIT | Optional Windows DML EP |
| `onnxruntime-gpu` | MIT | Optional CUDA EP |
| `torch` | BSD-3 | |
| `torch-directml` | BSD-3 | Windows DML only |
| `torchvision` | BSD-3 | |
| `nicegui` | MIT | Frontend server-rendered Vue3 |
| `websockets` | BSD-3 | |
| `ffmpeg-python` | Apache-2.0 | |
| `albumentations` | MIT | |
| `pycocotools` | BSD-2 / MIT | |
| `tqdm` | MIT / MPL-2.0 | |
| `tensorboard` | Apache-2.0 | |
| `jupyter`, `notebook` | BSD-3 | |
| `pytest`, `pytest-asyncio`, `pytest-cov` | MIT | |
| `ruff` | MIT | |
| `mypy` | MIT | |
| `pyinstaller` | GPL-2.0 with special bootloader exception | Build-time only, not redistributed to end users |

## 3. Frontend Stack

| Asset | Licence | Source / Notes |
|---|---|---|
| Tailwind CSS (precompiled, purged) | MIT | Bundled at `frontend/styles/tailwind.gen.css` (build artifact) |
| Quasar (via NiceGUI built-ins) | MIT | Loaded as ESM via `nicegui`; we do not fork |
| Vue 3 (NiceGUI runtime) | MIT | Server-rendered by `nicegui`, not bundled separately |
| 5 theme CSS files (this repo) | Apache-2.0 | `frontend/styles/themes/*.css` |
| Lato typeface (300/400/700, latin + latin-ext) | SIL OFL 1.1 | Self-hosted woff2 in `frontend/styles/fonts/`; full licence text bundled as `frontend/styles/fonts/OFL.txt`. Loaded via `frontend/styles/fonts.css`; no CDN request is made. |

## 4. System Binaries

| Binary | Licence | Bundled? |
|---|---|---|
| Python 3.10–3.12 runtime | PSF | Embedded by PyInstaller (one-dir build) |
| FFmpeg (Windows build) | LGPL-2.1 / GPL-2.0 (depends on build) | Downloaded at install time via `scripts/postinstall.bat`, not bundled. LGPL build preferred. |
| Microsoft Visual C++ Redistributable | EULA | Downloaded at install time |

## 5. Explicitly Excluded

The following are **NOT** permitted in the deliverable and must not be
added without an architectural review:

- **Ultralytics YOLO** (AGPL-3.0) — incompatible with the proprietary
  distribution model. The training pipeline uses a clean
  YOLOX-style architecture under Apache-2.0.
- **Any GPL library statically linked into the installer** (the LGPL
  FFmpeg is the boundary; PyInstaller's bootloader is exempt under
  its special exception).

## 6. Updates

When `pyproject.toml` changes, run:

```bash
make dist-clean
pip install -e ".[dev]"
python -m scripts.licence_audit --out LICENCE.md
```

Reviewers: confirm every new dep has a compatible licence, update the
table above, and add a CHANGELOG entry.
