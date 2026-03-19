import os
import gzip
import shutil
import tempfile
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

AID_PUBLIC_URL = "https://ftp.nhc.noaa.gov/atcf/aid_public/"
OUTPUT_DIR = "/data/gempak/atcf/"
DOWNLOAD_DIR = "./"
CHUNK_SIZE = 64 * 1024  # 64KB


def get_file_links(session):
    """Parse the HTML directory listing and return a list of .dat.gz filenames."""
    r = session.get(AID_PUBLIC_URL, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = [a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".dat.gz")]
    return links


def get_file_modtime(session, filename):
    """Get the modification time of a file from the HTTP headers (if available)."""
    url = AID_PUBLIC_URL + filename
    r = session.head(url, allow_redirects=True, timeout=20)
    if "Last-Modified" in r.headers:
        # parsedate_to_datetime handles timezone-aware parsing reliably
        try:
            dt = parsedate_to_datetime(r.headers["Last-Modified"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None
    return None


def download_file(session, url, dest_path, chunk_size=CHUNK_SIZE):
    """Stream-download to a temp file and atomically move to dest_path."""
    tmp_dir = os.path.dirname(dest_path) or "."
    tmp = None
    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False) as tmp:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        tmp.write(chunk)
                tmp.flush()
        # atomic replace
        os.replace(tmp.name, dest_path)
        tmp = None
    finally:
        # remove incomplete temp file if something went wrong
        if tmp is not None and os.path.exists(tmp.name):
            try:
                os.remove(tmp.name)
            except Exception:
                pass


def decompress_gz(src_path, dest_path, buffer_size=CHUNK_SIZE):
    """Decompress gzip file using buffered copy to avoid loading whole file into memory."""
    with gzip.open(src_path, "rb") as f_in, open(dest_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=buffer_size)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main(min_age_hours=0):
    ensure_dir(DOWNLOAD_DIR)
    ensure_dir(OUTPUT_DIR)
    now = datetime.now(timezone.utc)

    with requests.Session() as session:
        files = get_file_links(session)
        print(f"Found {len(files)} .dat.gz files.")

        for filename in files:
            url = AID_PUBLIC_URL + filename
            download_path = os.path.join(DOWNLOAD_DIR, filename)
            decompressed_name = filename[:-3]  # Remove '.gz'
            decompressed_path = os.path.join(OUTPUT_DIR, decompressed_name)

            # Age filter (if specified)
            if min_age_hours > 0:
                modtime = get_file_modtime(session, filename)
                if modtime:
                    age_hours = (now - modtime).total_seconds() / 3600
                    if age_hours < min_age_hours:
                        print(f"Skipping {filename}: age {age_hours:.1f}h < min_age {min_age_hours}h")
                        continue

            print(f"Downloading {filename} ...")
            download_file(session, url, download_path)
            print(f"Decompressing {filename} ...")
            decompress_gz(download_path, decompressed_path)
            print(f"Decompressed to {decompressed_path}")
            try:
                os.remove(download_path)  # Delete the .gz file after decompression
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    # Set min_age_hours as desired (e.g., 0 means download all)
    main(min_age_hours=0)
