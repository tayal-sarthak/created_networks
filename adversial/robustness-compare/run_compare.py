
import os
import sys
import subprocess
import tarfile
from pathlib import Path

DEFAULT_MODELS = ["resnet18"]


def pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _venv_python(script_dir: Path) -> str:
    venv_py = script_dir / "venv" / "bin" / "python3"
    if venv_py.exists():
        return venv_py.as_posix()
    return sys.executable


def _extract_blur_tar_if_present(script_dir: Path, imagenet_c_path: Path) -> None:
    blur_tar = script_dir / "blur.tar"
    if not blur_tar.exists():
        return
    imagenet_c_path.mkdir(parents=True, exist_ok=True)
    blur_markers = [
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "gaussian_blur",
    ]
    already_extracted = False
    for d in blur_markers:
        if (imagenet_c_path / d).exists():
            already_extracted = True
            break
    if already_extracted:
        return
    try:
        with tarfile.open(blur_tar.as_posix(), "r") as tf:
            tf.extractall(path=imagenet_c_path.as_posix())
        print(f"extracted {blur_tar.name} -> {imagenet_c_path}")
    except Exception as e:
        raise SystemExit(f"failed to extract {blur_tar}: {e}")


def _maybe_use_nested_blur(imagenet_c_path: Path) -> Path:
    nested = imagenet_c_path / "blur"
    if not nested.exists():
        return imagenet_c_path
    markers = [
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "gaussian_blur",
    ]
    has_markers = False
    for d in markers:
        if (nested / d).exists():
            has_markers = True
            break
    if has_markers:
        return nested
    return imagenet_c_path


def _looks_like_imagenet_c(path: Path) -> bool:
    candidates = [
        "gaussian_noise",
        "shot_noise",
        "defocus_blur",
        "contrast",
        "jpeg_compression",
        "motion_blur",
        "zoom_blur",
        "glass_blur",
        "gaussian_blur",
    ]
    for name in candidates:
        if (path / name).exists():
            return True
    return False


def _must_exist(p: str, label: str) -> None:
    if not p:
        raise SystemExit(f"missing {label}")
    if not Path(p).exists():
        raise SystemExit(f"{label} not found: {p}")


def _available_corruptions(root: Path, candidates: list[str]) -> list[str]:
    found = []
    for name in candidates:
        if (root / name).exists():
            found.append(name)
    return found


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    compare_script = script_dir / "compare_models.py"
    if not compare_script.exists():
        raise SystemExit(f"compare_models.py not found next to this script: {compare_script}")

    imagenet_c = os.environ.get("IMAGENET_C")
    if not imagenet_c:
        imagenet_c = (repo_root / "ImageNet-C").as_posix()
    # clean val is optional; only require it if the user explicitly set IMAGENET_VAL
    imagenet_val = os.environ.get("IMAGENET_VAL")
    val_env_set = "IMAGENET_VAL" in os.environ
    if not imagenet_val:
        imagenet_val = (repo_root / "imagenet" / "val").as_posix()
    imagenet_p = os.environ.get("IMAGENET_P", "")

    inc_path = Path(imagenet_c)
    _extract_blur_tar_if_present(script_dir, inc_path)
    inc_path = _maybe_use_nested_blur(inc_path)
    imagenet_c = inc_path.as_posix()

    _must_exist(imagenet_c, "imagenet-c root folder")
    if not _looks_like_imagenet_c(Path(imagenet_c)):
        names = []
        try:
            for p in Path(imagenet_c).iterdir():
                names.append(p.name)
        except Exception:
            names = []
        found = ", ".join(names) if names else "(no entries)"
        raise SystemExit(
            f"imagenet-c folder does not look like dataset: {imagenet_c}\n"
            f"expected subfolders like 'defocus_blur', 'motion_blur', ...\n"
            f"found: {found}"
        )
    if imagenet_p:
        _must_exist(imagenet_p, "imagenet-p root folder")
    if imagenet_val and Path(imagenet_val).as_posix().strip():
        if not Path(imagenet_val).exists():
            if val_env_set:
                raise SystemExit(f"imagenet val folder not found: {imagenet_val}")
            else:
                # no env var set so skip clean eval
                imagenet_val = ""

    py = _venv_python(script_dir)
    device = pick_device()
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    command = []
    command.append(py)
    command.append(str(compare_script))
    command.append("--models")
    models = DEFAULT_MODELS
    for m in models:
        command.append(m)
    command.append("--imagenet-c")
    command.append(imagenet_c)
    command.append("--device")
    command.append(device)
    command.append("--output-json")
    command.append(str(script_dir / "results.json"))
    command.append("--severities")
    command.append("1")
    command.append("2")
    command.append("3")
    command.append("4")
    command.append("5")
    command.append("--max-per-class")
    command.append("12")
    command.append("--num-workers")
    command.append("0")
    # pass only the blur corruptions that actually exist to avoid noisy "skip" lines in blur.tar! 
    # remember, only blur.tar! 
    blur_candidates = [
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "gaussian_blur",
    ]
    present_corruptions = _available_corruptions(Path(imagenet_c), blur_candidates)
    if present_corruptions:
        command.append("--corruptions")
        for c in present_corruptions:
            command.append(c)
    if device == "cpu":
        command.append("--batch-size")
        command.append("64")
    elif device == "mps":
        command.append("--batch-size")
        command.append("128")
    else:
        command.append("--batch-size")
        command.append("128")
    if imagenet_val and Path(imagenet_val).exists():
        command.append("--imagenet-val")
        command.append(imagenet_val)
    if imagenet_p:
        command.append("--imagenet-p")
        command.append(imagenet_p)

    print("running compare")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"error: {e}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
