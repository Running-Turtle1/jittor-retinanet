import os
import requests
from tqdm import tqdm
import zipfile

urls = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

base_dir = "pytorch-retinanet/coco"
images_dir = os.path.join(base_dir, "images")
annotations_dir = os.path.join(base_dir, "annotations")

os.makedirs(images_dir, exist_ok = True)
os.makedirs(annotations_dir, exist_ok = True)


def download_file(url, save_path):
    response = requests.get(url, stream = True)
    total_size = int(response.headers.get('content-length', 0))

    with open(save_path, "wb") as file, tqdm(
            desc = save_path,
            total = total_size,
            unit = 'B',
            unit_scale = True,
            ncols = 100
    ) as bar:
        for data in response.iter_content(chunk_size = 1024):
            bar.update(len(data))
            file.write(data)


def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"{zip_path} extraction completed!")


for name, url in urls.items():
    print(f"Downloading {name}...")

    if name == "annotations":
        save_path = os.path.join(annotations_dir, f"{name}.zip")
        extract_dir = annotations_dir
    else:
        save_path = os.path.join(images_dir, f"{name}.zip")
        extract_dir = os.path.join(images_dir, name)

    download_file(url, save_path)
    extract_zip(save_path, extract_dir)

    print(f"{name} download and extraction completed!")

print("All files downloaded and extracted!")
