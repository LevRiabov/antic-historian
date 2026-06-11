"""Async corpus downloader: manifest entries -> corpus/raw/pg<ID>.txt.

Idempotent: existing files are skipped unless force=True. Concurrency is kept
low (3) to be polite to Project Gutenberg.
"""

import asyncio
from pathlib import Path
from typing import Literal

import httpx

from ahx.ingest.manifest import ManifestEntry

DownloadStatus = Literal["downloaded", "cached", "failed"]


async def _download_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    entry: ManifestEntry,
    dest_dir: Path,
    force: bool,
) -> tuple[ManifestEntry, DownloadStatus, str]:
    dest = dest_dir / entry.raw_filename
    if dest.exists() and not force:
        return entry, "cached", ""
    async with semaphore:
        try:
            response = await client.get(entry.txt_url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return entry, "failed", str(exc)
    dest.write_text(response.text, encoding="utf-8")
    return entry, "downloaded", ""


async def download_all(
    entries: list[ManifestEntry],
    dest_dir: Path,
    force: bool = False,
    concurrency: int = 3,
) -> list[tuple[ManifestEntry, DownloadStatus, str]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        tasks = [_download_one(client, semaphore, entry, dest_dir, force) for entry in entries]
        return list(await asyncio.gather(*tasks))
