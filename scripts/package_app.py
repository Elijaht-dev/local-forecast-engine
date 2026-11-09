"""Build a source bundle for installation through HA OS Local apps."""

from io import BytesIO
from pathlib import Path
from tarfile import TarInfo
from tarfile import open as open_tar

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "local_forecast"
OUTPUT = ROOT / "dist" / "local-forecast-app.tar.gz"


def package_app() -> Path:
    OUTPUT.parent.mkdir(exist_ok=True)
    paths = [
        ROOT / name
        for name in ("Dockerfile", ".dockerignore", "LICENSE", "constraints.txt", "pyproject.toml")
    ]
    paths.extend(sorted((ROOT / "local_forecast_engine").glob("*.py")))
    config = (APP / "config.yaml").read_text(encoding="utf-8")
    local_config = (
        "\n".join(line for line in config.splitlines() if not line.startswith("image:")) + "\n"
    )

    with open_tar(OUTPUT, "w:gz") as bundle:
        for path in paths:
            bundle.add(path, arcname=Path("local_forecast") / path.relative_to(ROOT))
        for path in sorted(APP.iterdir()):
            if path.is_file() and path.name != "config.yaml":
                bundle.add(path, arcname=Path("local_forecast") / path.name)
        body = local_config.encode("utf-8")
        entry = TarInfo("local_forecast/config.yaml")
        entry.size = len(body)
        bundle.addfile(entry, BytesIO(body))
    return OUTPUT


if __name__ == "__main__":
    print(package_app())
