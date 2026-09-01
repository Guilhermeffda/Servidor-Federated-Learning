#!/usr/bin/env python3
"""Acquire and safely extract the official TACO archive from Zenodo."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import logging
import os
import re
import shutil
import socket
import stat
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LOGGER = logging.getLogger("taco-download")
DATASET_NAME = "TACO - Trash Annotations in Context"
GITHUB_URL = "https://github.com/pedropro/TACO"
MAP_10_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/detector/taco_config/map_10.csv"
MAP_10_NAME = "map_10.csv"
ZENODO_RECORD = "3587843"
ZENODO_DOI = "10.5281/zenodo.3587843"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
ZENODO_RECORD_URL = f"https://zenodo.org/records/{ZENODO_RECORD}"
ARCHIVE_NAME = "TACO.zip"
EXPECTED_MD5 = "e9149407d883e21a8d224feef8210920"
USER_AGENT = "taco-data-acquisition/2.0"
TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}
CHUNK_SIZE = 1024 * 1024


class PipelineError(RuntimeError):
    """Indicate an acquisition error that should produce a clean exit."""


class IncompleteDownloadError(RuntimeError):
    """Indicate that a response ended before the expected archive size."""


@dataclass(frozen=True)
class ArchiveMetadata:
    """Validated metadata for the selected Zenodo archive."""

    name: str
    size_bytes: int | None
    checksum: str
    download_url: str


@dataclass(frozen=True)
class ArchiveState:
    """Observed state of the local archive after acquisition."""

    local_size_bytes: int | None
    observed_md5: str | None
    observed_sha256: str | None
    integrity_status: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the deliberately small command-line interface."""
    parser = argparse.ArgumentParser(
        description="Download and extract the official TACO archive from Zenodo."
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="validate Zenodo metadata without downloading the archive",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="network timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="retries after transient network failures (default: 2)",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    return args


def utc_now() -> str:
    """Return an ISO 8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def checksum_file(path: Path) -> tuple[str, str]:
    """Calculate MD5 and SHA-256 in one streaming pass."""
    md5_digest = hashlib.md5(usedforsecurity=False)
    sha256_digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
                md5_digest.update(chunk)
                sha256_digest.update(chunk)
    except OSError as exc:
        raise PipelineError(f"could not read archive {path}: {exc}") from exc
    return md5_digest.hexdigest(), sha256_digest.hexdigest()


def is_transient_network_error(exc: BaseException) -> bool:
    """Return whether retrying a network exception can reasonably help."""
    if isinstance(exc, HTTPError):
        return exc.code in TRANSIENT_HTTP_CODES
    return isinstance(
        exc,
        (
            URLError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            http.client.IncompleteRead,
            IncompleteDownloadError,
        ),
    )


def fetch_json(url: str, timeout: float, retries: int) -> dict[str, Any]:
    """Fetch a JSON object with bounded retries."""
    for attempt in range(retries + 1):
        try:
            request = Request(
                url,
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            )
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise PipelineError("Zenodo metadata response is not a JSON object")
            return payload
        except (
            HTTPError,
            URLError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            http.client.IncompleteRead,
        ) as exc:
            if is_transient_network_error(exc) and attempt < retries:
                LOGGER.warning("Metadata request failed; retrying: %s", exc)
                time.sleep(min(2**attempt, 4))
                continue
            raise PipelineError(f"could not obtain Zenodo metadata: {exc}") from exc
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise PipelineError(f"Zenodo returned invalid JSON: {exc}") from exc
    raise AssertionError("metadata retry loop ended unexpectedly")


def normalize_md5(checksum: Any) -> str:
    """Normalize a Zenodo MD5 value and reject unknown formats."""
    if not isinstance(checksum, str):
        raise PipelineError("TACO.zip metadata does not contain a checksum")
    normalized = checksum.lower()
    if normalized.startswith("md5:"):
        normalized = normalized[4:]
    if not re.fullmatch(r"[0-9a-f]{32}", normalized):
        raise PipelineError(f"unsupported TACO.zip checksum: {checksum!r}")
    return normalized


def select_download_url(file_metadata: dict[str, Any]) -> str:
    """Select the official archive URL exposed by Zenodo metadata."""
    links = file_metadata.get("links")
    if not isinstance(links, dict):
        raise PipelineError("TACO.zip metadata does not contain download links")
    for key in ("content", "download", "self"):
        candidate = links.get(key)
        if isinstance(candidate, str) and candidate.startswith("https://"):
            return candidate
    raise PipelineError("TACO.zip metadata has no supported HTTPS download URL")


def validate_metadata(document: dict[str, Any]) -> ArchiveMetadata:
    """Validate the expected record and locate exactly TACO.zip."""
    if str(document.get("id")) != ZENODO_RECORD:
        raise PipelineError(
            f"unexpected Zenodo record: {document.get('id')!r}; "
            f"expected {ZENODO_RECORD}"
        )
    files = document.get("files")
    if not isinstance(files, list):
        raise PipelineError("Zenodo record does not contain a files collection")
    matches = [
        item
        for item in files
        if isinstance(item, dict) and item.get("key") == ARCHIVE_NAME
    ]
    if len(matches) != 1:
        raise PipelineError(
            f"expected exactly one {ARCHIVE_NAME} in Zenodo metadata; "
            f"found {len(matches)}"
        )

    selected = matches[0]
    observed_md5 = normalize_md5(selected.get("checksum"))
    if observed_md5 != EXPECTED_MD5:
        raise PipelineError(
            "Zenodo checksum divergence: "
            f"expected {EXPECTED_MD5}, received {observed_md5}; download aborted"
        )
    size = selected.get("size")
    if size is not None and (not isinstance(size, int) or size <= 0):
        raise PipelineError(f"invalid TACO.zip size in metadata: {size!r}")
    return ArchiveMetadata(
        name=ARCHIVE_NAME,
        size_bytes=size,
        checksum=observed_md5,
        download_url=select_download_url(selected),
    )


def parse_content_range(value: str | None) -> tuple[int, int, int | None]:
    """Parse a byte Content-Range header."""
    if value is None:
        raise PipelineError("partial response is missing Content-Range")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", value.strip())
    if match is None:
        raise PipelineError(f"invalid Content-Range: {value!r}")
    start, end = int(match.group(1)), int(match.group(2))
    total = None if match.group(3) == "*" else int(match.group(3))
    if end < start:
        raise PipelineError(f"invalid Content-Range bounds: {value!r}")
    return start, end, total


def copy_response(response: BinaryIO, part_path: Path, mode: str) -> None:
    """Stream an HTTP response to the partial archive."""
    try:
        output = part_path.open(mode)
    except OSError as exc:
        raise PipelineError(f"could not open partial archive: {exc}") from exc
    with output:
        while chunk := response.read(CHUNK_SIZE):
            try:
                output.write(chunk)
            except OSError as exc:
                raise PipelineError(
                    f"could not write partial archive: {exc}"
                ) from exc


def download_attempt(
    metadata: ArchiveMetadata, part_path: Path, timeout: float
) -> None:
    """Perform one full or resumed HTTP download attempt."""
    try:
        offset = part_path.stat().st_size if part_path.exists() else 0
    except OSError as exc:
        raise PipelineError(f"could not inspect partial archive: {exc}") from exc
    if metadata.size_bytes is not None and offset > metadata.size_bytes:
        raise PipelineError(
            f"partial archive is larger than expected ({offset} > "
            f"{metadata.size_bytes}); it was preserved for inspection"
        )
    if metadata.size_bytes is not None and offset == metadata.size_bytes:
        LOGGER.info("Partial archive already has the expected size")
        return

    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        LOGGER.info("Resuming archive download at byte %d", offset)
    request = Request(metadata.download_url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        status = response.getcode()
        mode = "wb"
        if offset and status == 206:
            start, _, total = parse_content_range(
                response.headers.get("Content-Range")
            )
            if start != offset:
                raise PipelineError(
                    f"partial response starts at {start}, expected {offset}"
                )
            if (
                total is not None
                and metadata.size_bytes is not None
                and total != metadata.size_bytes
            ):
                raise PipelineError(
                    f"Content-Range total {total} differs from metadata size "
                    f"{metadata.size_bytes}"
                )
            mode = "ab"
        elif offset and status == 200:
            LOGGER.warning("Server ignored Range; safely restarting partial file")
        elif status == 206:
            start, _, _ = parse_content_range(response.headers.get("Content-Range"))
            if start != 0:
                raise PipelineError(
                    f"initial partial response starts at unexpected byte {start}"
                )
        elif status != 200:
            raise PipelineError(f"unexpected archive HTTP status: {status}")
        copy_response(response, part_path, mode)

    actual_size = part_path.stat().st_size
    if metadata.size_bytes is not None and actual_size < metadata.size_bytes:
        raise IncompleteDownloadError(
            f"archive is incomplete ({actual_size}/{metadata.size_bytes} bytes)"
        )
    if metadata.size_bytes is not None and actual_size > metadata.size_bytes:
        raise PipelineError(
            f"archive exceeds expected size ({actual_size}/{metadata.size_bytes})"
        )


def acquire_archive(
    metadata: ArchiveMetadata,
    archive_path: Path,
    timeout: float,
    retries: int,
) -> ArchiveState:
    """Download, validate, and atomically promote the official archive."""
    if archive_path.exists():
        if not archive_path.is_file():
            raise PipelineError(f"archive destination is not a file: {archive_path}")
        observed_md5, observed_sha256 = checksum_file(archive_path)
        if observed_md5 != EXPECTED_MD5:
            raise PipelineError(
                f"existing archive has invalid MD5 {observed_md5}; expected "
                f"{EXPECTED_MD5}. The file was not modified"
            )
        LOGGER.info("Reusing existing archive with verified MD5")
        return ArchiveState(
            archive_path.stat().st_size,
            observed_md5,
            observed_sha256,
            "verified",
        )

    try:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PipelineError(f"could not create archives directory: {exc}") from exc
    part_path = archive_path.with_name(archive_path.name + ".part")
    for attempt in range(retries + 1):
        try:
            download_attempt(metadata, part_path, timeout)
            break
        except (
            HTTPError,
            URLError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            http.client.IncompleteRead,
            IncompleteDownloadError,
        ) as exc:
            if is_transient_network_error(exc) and attempt < retries:
                LOGGER.warning("Archive download interrupted; retrying: %s", exc)
                time.sleep(min(2**attempt, 4))
                continue
            raise PipelineError(
                f"archive download failed; partial data was preserved: {exc}"
            ) from exc

    observed_md5, observed_sha256 = checksum_file(part_path)
    if observed_md5 != EXPECTED_MD5:
        raise PipelineError(
            f"downloaded archive has invalid MD5 {observed_md5}; expected "
            f"{EXPECTED_MD5}. Partial file was preserved"
        )
    if archive_path.exists():
        raise PipelineError(
            "archive destination appeared during download; no file was overwritten"
        )
    try:
        os.replace(part_path, archive_path)
    except OSError as exc:
        raise PipelineError(f"could not promote verified archive: {exc}") from exc
    return ArchiveState(
        archive_path.stat().st_size,
        observed_md5,
        observed_sha256,
        "verified",
    )


def safe_zip_destination(extraction_root: Path, member_name: str) -> Path:
    """Resolve one ZIP member below extraction_root or reject it."""
    if not member_name or "\x00" in member_name:
        raise PipelineError("ZIP contains an empty or invalid member name")
    posix_name = PurePosixPath(member_name)
    windows_name = PureWindowsPath(member_name)
    if (
        posix_name.is_absolute()
        or windows_name.is_absolute()
        or windows_name.drive
    ):
        raise PipelineError(f"ZIP contains an absolute path: {member_name!r}")
    if ".." in posix_name.parts or ".." in windows_name.parts:
        raise PipelineError(f"ZIP contains path traversal: {member_name!r}")
    destination = (extraction_root / Path(member_name)).resolve()
    try:
        destination.relative_to(extraction_root.resolve())
    except ValueError as exc:
        raise PipelineError(
            f"ZIP member escapes extraction root: {member_name!r}"
        ) from exc
    return destination


def validated_zip_members(
    archive: zipfile.ZipFile, extraction_root: Path
) -> list[tuple[zipfile.ZipInfo, Path]]:
    """Validate every member before extraction writes any content."""
    validated: list[tuple[zipfile.ZipInfo, Path]] = []
    for member in archive.infolist():
        destination = safe_zip_destination(extraction_root, member.filename)
        unix_mode = member.external_attr >> 16
        if unix_mode and stat.S_ISLNK(unix_mode):
            raise PipelineError(
                f"ZIP symbolic links are not supported: {member.filename!r}"
            )
        validated.append((member, destination))
    return validated


def marker_matches(marker_path: Path, archive_sha256: str) -> bool:
    """Return whether the completion marker matches the verified archive."""
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(marker, dict)
        and marker.get("archive_name") == ARCHIVE_NAME
        and marker.get("archive_sha256") == archive_sha256
        and marker.get("status") == "completed"
    )


def write_json_atomic(path: Path, document: Any) -> None:
    """Write deterministic, readable JSON through a temporary file."""
    temporary = path.with_name(path.name + ".part")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(
                document,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise PipelineError(f"could not write JSON file {path}: {exc}") from exc


def extract_archive(
    archive_path: Path,
    extraction_root: Path,
    marker_path: Path,
    archive_sha256: str,
) -> str:
    """Safely extract a verified ZIP and write a completion marker."""
    if extraction_root.is_dir() and marker_matches(marker_path, archive_sha256):
        LOGGER.info("Reusing extraction verified by completion marker")
        return "completed"
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = validated_zip_members(archive, extraction_root)
            extraction_root.mkdir(parents=True, exist_ok=True)
            for member, destination in members:
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".part")
                with archive.open(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=CHUNK_SIZE)
                os.replace(temporary, destination)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise PipelineError(f"archive extraction failed: {exc}") from exc

    write_json_atomic(
        marker_path,
        {
            "archive_name": ARCHIVE_NAME,
            "archive_sha256": archive_sha256,
            "completed_at_utc": utc_now(),
            "status": "completed",
        },
    )
    return "completed"


def download_map_10(output_path: Path, timeout: float, retries: int) -> None:
    """Download the official TACO-10 category mapping."""
    if output_path.is_file():
        LOGGER.info("Reusing existing TACO-10 mapping: %s", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries + 1):
        try:
            request = Request(
                MAP_10_URL,
                headers={"User-Agent": USER_AGENT},
            )
            with urlopen(request, timeout=timeout) as response:
                content = response.read()

            if not content.strip():
                raise PipelineError("downloaded map_10.csv is empty")

            output_path.write_bytes(content)
            LOGGER.info("TACO-10 mapping downloaded: %s", output_path)
            return

        except (HTTPError, URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
            if is_transient_network_error(exc) and attempt < retries:
                LOGGER.warning("TACO-10 mapping download failed; retrying: %s", exc)
                time.sleep(min(2**attempt, 4))
                continue
            raise PipelineError(f"could not download {MAP_10_NAME}: {exc}") from exc
        except OSError as exc:
            raise PipelineError(f"could not save {MAP_10_NAME}: {exc}") from exc


def extraction_status(extraction_root: Path, marker_path: Path) -> str:
    """Describe extraction state without reading the multi-gigabyte archive."""
    if marker_path.is_file():
        return "completed_marker_present"
    if extraction_root.exists():
        return "incomplete"
    return "not_started"


def preserve_legacy_manifest(manifest_path: Path) -> Path | None:
    """Preserve one old Flickr manifest before replacing it."""
    if not manifest_path.is_file():
        return None
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PipelineError(f"could not inspect existing manifest: {exc}") from exc
    if isinstance(document, dict) and document.get("zenodo_record") == ZENODO_RECORD:
        return None
    legacy_path = manifest_path.with_name("download_manifest_flickr_legacy.json")
    if not legacy_path.exists():
        try:
            os.replace(manifest_path, legacy_path)
        except OSError as exc:
            raise PipelineError(f"could not preserve legacy manifest: {exc}") from exc
        LOGGER.info("Preserved old Flickr manifest as %s", legacy_path.name)
        return legacy_path
    LOGGER.info("A legacy Flickr manifest is already preserved")
    return legacy_path


def build_manifest(
    metadata: ArchiveMetadata,
    archive_state: ArchiveState,
    extraction_state: str,
) -> dict[str, Any]:
    """Build the archive-level acquisition manifest."""
    return {
        "archive": asdict(metadata),
        "dataset": DATASET_NAME,
        "dataset_repository": GITHUB_URL,
        "doi": ZENODO_DOI,
        "extraction_status": extraction_state,
        "integrity_status": archive_state.integrity_status,
        "local_size_bytes": archive_state.local_size_bytes,
        "md5_expected": EXPECTED_MD5,
        "md5_observed": archive_state.observed_md5,
        "official_source": ZENODO_RECORD_URL,
        "sha256_observed": archive_state.observed_sha256,
        "timestamp_utc": utc_now(),
        "url_used": metadata.download_url,
        "zenodo_record": ZENODO_RECORD,
    }


def run(args: argparse.Namespace) -> int:
    """Run metadata validation and, unless requested otherwise, acquisition."""
    project_root = Path(__file__).resolve().parent.parent
    raw_root = project_root / "data" / "raw" / "taco"
    archive_path = raw_root / "archives" / ARCHIVE_NAME
    extraction_root = raw_root / "extracted"
    metadata_root = raw_root / "metadata"
    manifest_path = metadata_root / "download_manifest.json"
    marker_path = metadata_root / "extraction_complete.json"
    map_10_path = extraction_root / "TACO" / "detector" / "taco_config" / MAP_10_NAME

    metadata = validate_metadata(
        fetch_json(ZENODO_API_URL, args.timeout, args.retries)
    )
    LOGGER.info(
        "Validated Zenodo record %s: %s, %s bytes, MD5 %s",
        ZENODO_RECORD,
        metadata.name,
        metadata.size_bytes if metadata.size_bytes is not None else "unknown",
        metadata.checksum,
    )
    preserve_legacy_manifest(manifest_path)

    if args.metadata_only:
        local_size = archive_path.stat().st_size if archive_path.is_file() else None
        archive_state = ArchiveState(
            local_size,
            None,
            None,
            "not_checked_metadata_only" if local_size is not None else "not_downloaded",
        )
        extraction_state = extraction_status(extraction_root, marker_path)
        write_json_atomic(
            manifest_path,
            build_manifest(metadata, archive_state, extraction_state),
        )
        print("Zenodo metadata validated; archive download was not started.")
        return 0

    archive_state = acquire_archive(
        metadata, archive_path, args.timeout, args.retries
    )
    if archive_state.observed_sha256 is None:
        raise PipelineError("verified archive is missing its SHA-256 digest")
    extraction_state = extract_archive(
        archive_path,
        extraction_root,
        marker_path,
        archive_state.observed_sha256,
    )
    download_map_10(map_10_path, args.timeout, args.retries)
    write_json_atomic(
        manifest_path,
        build_manifest(metadata, archive_state, extraction_state),
    )
    print("TACO archive verified and extraction completed.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Configure logging and convert expected failures to a nonzero exit."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return run(parse_args(argv))
    except PipelineError as exc:
        LOGGER.error("%s", exc)
        return 1
    except OSError as exc:
        LOGGER.error("local filesystem error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
