from _future_ import annotations
from typing import Iterable, List, Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse
import mimetypes
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # ✅ progress bar
import os


# -----------------------
# Filesystem helpers
# -----------------------

def ensure_dir(path: Path | str) -> Path:
    """Create a directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize_filename(name: str) -> str:
    """Replace characters unsafe for most filesystems and normalize whitespace."""
    bad = '<>:"/\\|?*'
    for ch in bad:
        name = name.replace(ch, "_")
    name = "_".join(name.split())
    return name.strip() or "file"


# -----------------------
# HTTP session with retry
# -----------------------

def build_session(
    total_retries: int = 5,
    backoff_factor: float = 0.5,
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
    pool_connections: int = 100,
    pool_maxsize: int = 100,
    user_agent: str = "image-downloader/1.0",
) -> requests.Session:
    """Build a requests.Session with retry and connection pooling."""
    retry = Retry(
        total=total_retries,
        connect=3,
        read=3,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "OPTIONS", "TRACE"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_connections, pool_maxsize=pool_maxsize)
    s = requests.Session()
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": user_agent})
    return s


# -----------------------
# Download helpers
# -----------------------

def _guess_ext_from_url_or_type(url_path: str, content_type: Optional[str]) -> str:
    """Infer file extension from URL path or HTTP Content-Type."""
    suffix = Path(url_path).suffix.lower()
    if suffix and len(suffix) <= 5:
        return suffix

    if content_type:
        base_ct = content_type.split(";")[0].strip()
        ext = mimetypes.guess_extension(base_ct) or ".jpg"
        if ext in (".jpe", ".jpeg"):
            return ".jpg"
        if ext == ".htm":
            return ".jpg"
        return ext
    return ".jpg"


def log_failure(log_file: Path | str, url: str, sample_id, error: str, out_dir: Path):
    """Write a failure entry to the specified log file (outside image folder)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] URL: {url} | sample_id: {sample_id} | Error: {error} | Output Dir: {out_dir}\n"

    log_file = Path(log_file)
    ensure_dir(log_file.parent)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)


def download_one(
    session: requests.Session,
    url: str,
    dest_dir: Path | str,
    filename: str,
    timeout: Tuple[int, int] = (10, 60),
) -> Tuple[str, Optional[str], Optional[str]]:
    """Download a single image and return (url, path, error)."""
    dest_dir = ensure_dir(dest_dir)
    parsed = urlparse(url)
    ext = _guess_ext_from_url_or_type(parsed.path, None)
    out_path = Path(dest_dir) / f"{filename}{ext}"

    try:
        with session.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()

            if "Content-Type" in resp.headers and ext in ["", ".jpg"]:
                refined_ext = _guess_ext_from_url_or_type(parsed.path, resp.headers.get("Content-Type"))
                if refined_ext and refined_ext != ext:
                    out_path = Path(dest_dir) / f"{filename}{refined_ext}"

            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return (url, str(out_path), None)
    except Exception as e:
        return (url, None, str(e))


def download_from_df(
    df,
    column: str,
    out_dir: Path | str,
    base_name: str,
    log_file: str,
    id_column: str,
    max_workers: int = 16,
    session: Optional[requests.Session] = None,
) -> dict:
    """Download images from a DataFrame column with logging of failures and tqdm progress."""
    out_dir = ensure_dir(out_dir)
    session = session or build_session()

    urls = []
    for _, row in df.iterrows():
        url = str(row[column]).strip()
        if url:
            sample_id = row[id_column]
            filename = f"{base_name}_{sample_id}"
            urls.append((url, filename, sample_id))

    ok = 0
    fail = 0
    results = []

    log_path = Path(log_file)
    ensure_dir(log_path.parent)

    total = len(urls)
    if total == 0:
        print(f"No URLs found in column '{column}'.")
        return {"ok": 0, "fail": 0, "results": []}

    print(f"\n🚀 Starting download for {total} {base_name} images...\n")

    # tqdm progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as ex, tqdm(
        total=total,
        desc=f"Downloading {base_name} images",
        unit="img",
        ncols=90,
        dynamic_ncols=True,
        leave=True,
        colour="cyan"
    ) as pbar:
        futures = [ex.submit(download_one, session, url, out_dir, filename) for url, filename, _ in urls]

        for (url, filename, sample_id), fut in zip(urls, as_completed(futures)):
            url, path, err = fut.result()
            if err:
                fail += 1
                log_failure(log_path, url, sample_id, err, out_dir)
            else:
                ok += 1
            results.append((url, path, err))
            pbar.update(1)

    print(f"\n✅ Download completed for {base_name}: {ok} succeeded, {fail} failed.")
    print(f"📄 Log file: {log_path}\n")

    return {"ok": ok, "fail": fail, "results": results}