import os
import zipfile
from pathlib import Path

import requests


def download_and_extract(
    url: str,
    extract_to: Path,
    chunk_size: int | None = 8192,
):
    """Download a ZIP file from a URL and extract its contents.

    Args:
        url: The URL of the ZIP file to download.
        extract_to: The directory where the contents should be extracted.
        chunk_size: The size of chunks to read when streaming. If None, the entire file will be read at once.
    """
    zip_path = os.path.basename(url.split("?")[0])
    stream = chunk_size is not None

    response = requests.get(url, stream=stream)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        if stream:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
        else:
            f.write(response.content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
