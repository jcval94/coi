"""Minimal PEP 517 backend without external build dependencies."""
from __future__ import annotations

import base64
import hashlib
import os
import re
import tarfile
import textwrap
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python <3.11
    import tomli as tomllib  # type: ignore

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "coi_fraud"


class ProjectMetadata(Dict[str, object]):
    name: str
    version: str
    summary: str
    requires_python: Optional[str]
    authors: List[Dict[str, str]]
    dependencies: List[str]
    readme_text: str


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name)


def _metadata() -> ProjectMetadata:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    readme = project.get("readme")
    readme_text = ""
    if isinstance(readme, dict):
        path = readme.get("path")
        if path:
            readme_text = (ROOT / path).read_text(encoding="utf-8")
    elif isinstance(readme, str):
        readme_text = (ROOT / readme).read_text(encoding="utf-8")

    return ProjectMetadata(
        name=project["name"],
        version=project["version"],
        summary=project.get("description", ""),
        requires_python=project.get("requires-python"),
        authors=project.get("authors", []),
        dependencies=project.get("dependencies", []),
        readme_text=readme_text,
    )


def get_requires_for_build_wheel(config_settings=None):  # noqa: D401 - required hook
    return []


def get_requires_for_build_editable(config_settings=None):  # noqa: D401 - required hook
    return []


def _metadata_lines(meta: ProjectMetadata) -> List[str]:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {meta['name']}",
        f"Version: {meta['version']}",
    ]
    summary = meta.get("summary")
    if summary:
        lines.append(f"Summary: {summary}")
    requires_python = meta.get("requires_python")
    if requires_python:
        lines.append(f"Requires-Python: {requires_python}")
    for author in meta.get("authors", []):
        name = author.get("name")
        email = author.get("email")
        if name and email:
            lines.append(f"Author: {name} <{email}>")
        elif name:
            lines.append(f"Author: {name}")
        elif email:
            lines.append(f"Author-email: {email}")
    for dep in meta.get("dependencies", []):
        lines.append(f"Requires-Dist: {dep}")
    if meta.get("readme_text"):
        lines.extend([
            "Description-Content-Type: text/markdown",
            "",
            meta["readme_text"],
        ])
    return lines


def _ensure_dist_info(meta: ProjectMetadata, destination: Path) -> Path:
    normalized = _normalize(meta["name"])
    dist_info = destination / f"{normalized}-{meta['version']}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)

    metadata_content = "\n".join(_metadata_lines(meta)) + "\n"
    (dist_info / "METADATA").write_text(metadata_content, encoding="utf-8")

    wheel_content = textwrap.dedent(
        """\
        Wheel-Version: 1.0
        Generator: packaging_backend
        Root-Is-Purelib: true
        Tag: py3-none-any
        """
    ).lstrip()
    (dist_info / "WHEEL").write_text(wheel_content, encoding="utf-8")

    (dist_info / "top_level.txt").write_text("coi_fraud\n", encoding="utf-8")

    return dist_info


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    meta = _metadata()
    dist_info = _ensure_dist_info(meta, Path(metadata_directory))
    return dist_info.name


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_sdist(sdist_directory, config_settings=None):
    meta = _metadata()
    normalized = _normalize(meta["name"]).replace("_", "-")
    base_name = f"{normalized}-{meta['version']}"
    target = Path(sdist_directory) / f"{base_name}.tar.gz"

    with tarfile.open(target, "w:gz") as tar:
        root_prefix = Path(base_name)
        for path in [ROOT / "pyproject.toml", ROOT / "packaging_backend.py", SRC_DIR]:
            if path.is_dir():
                for item in path.rglob("*"):
                    if item.is_file():
                        arcname = root_prefix / item.relative_to(ROOT)
                        tar.add(item, arcname=str(arcname))
            else:
                arcname = root_prefix / path.relative_to(ROOT)
                tar.add(path, arcname=str(arcname))
    return target.name


def _iter_package_files() -> Iterable[Path]:
    if not SRC_DIR.exists():
        return []
    for path in SRC_DIR.rglob("*"):
        if path.is_file():
            yield path


def _record_line(rel_path: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
    return f"{rel_path},sha256={digest},{len(data)}"


def _build_wheel_files(meta: ProjectMetadata, include_sources: bool, editable: bool) -> List[Tuple[str, bytes]]:
    files: List[Tuple[str, bytes]] = []
    normalized = _normalize(meta["name"])
    dist_info_name = f"{normalized}-{meta['version']}.dist-info"

    if include_sources:
        for path in _iter_package_files():
            rel = f"coi_fraud/{path.relative_to(SRC_DIR).as_posix()}"
            files.append((rel, path.read_bytes()))
    elif editable:
        pth_content = (str(ROOT) + os.linesep).encode("utf-8")
        files.append((f"{normalized}.pth", pth_content))

    metadata_content = "\n".join(_metadata_lines(meta)) + "\n"
    files.append((f"{dist_info_name}/METADATA", metadata_content.encode("utf-8")))

    wheel_content = textwrap.dedent(
        """\
        Wheel-Version: 1.0
        Generator: packaging_backend
        Root-Is-Purelib: true
        Tag: py3-none-any
        """
    ).lstrip()
    files.append((f"{dist_info_name}/WHEEL", wheel_content.encode("utf-8")))
    files.append((f"{dist_info_name}/top_level.txt", b"coi_fraud\n"))

    files.append((f"{dist_info_name}/RECORD", b""))
    return files


def _write_wheel(files: List[Tuple[str, bytes]], wheel_path: Path) -> str:
    record_entries: List[str] = []
    record_name = next(path for path, _ in files if path.endswith(".dist-info/RECORD"))
    with zipfile.ZipFile(wheel_path, "w") as zf:
        for path, data in files:
            if path == record_name:
                continue
            zf.writestr(path, data)
            record_entries.append(_record_line(path, data))
        record_content = "\n".join(record_entries + [f"{record_name},,"]) + "\n"
        zf.writestr(record_name, record_content.encode("utf-8"))
    return wheel_path.name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    meta = _metadata()
    normalized = _normalize(meta["name"])
    wheel_name = f"{normalized}-{meta['version']}-py3-none-any.whl"
    wheel_path = Path(wheel_directory) / wheel_name
    files = _build_wheel_files(meta, include_sources=True, editable=False)
    return _write_wheel(files, wheel_path)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    meta = _metadata()
    normalized = _normalize(meta["name"])
    wheel_name = f"{normalized}-{meta['version']}-py3-none-any.whl"
    wheel_path = Path(wheel_directory) / wheel_name
    files = _build_wheel_files(meta, include_sources=False, editable=True)
    return _write_wheel(files, wheel_path)
