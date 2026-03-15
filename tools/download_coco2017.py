import os
import zipfile

import requests
from tqdm import tqdm


URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(REPO_ROOT, "coco")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
ARCHIVE_DIR = os.path.join(BASE_DIR, "_archives")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)


def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(save_path, "wb") as file, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit="B",
        unit_scale=True,
        ncols=100,
    ) as bar:
        for data in response.iter_content(chunk_size=1024 * 1024):
            if data:
                file.write(data)
                bar.update(len(data))


def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"{zip_path} extraction completed.")


for name, url in URLS.items():
    print(f"Downloading {name}...")

    save_path = os.path.join(ARCHIVE_DIR, f"{name}.zip")
    extract_dir = BASE_DIR if name == "annotations" else IMAGES_DIR

    download_file(url, save_path)
    extract_zip(save_path, extract_dir)

    print(f"{name} download and extraction completed.")

print(f"All files downloaded and extracted under: {BASE_DIR}")
