import requests
import urllib.request
import progressbar
from bs4 import BeautifulSoup
from PIL import Image
from os import listdir
from os.path import isfile, join


def download_images(outdir):
    res = requests.get(
        "https://en.wikipedia.org/wiki/List_of_works_by_Vincent_van_Gogh")

    if int(res.status_code / 100) != 2:
        raise Exception("Error in receiving images")

    res = res.content

    soup = BeautifulSoup(res, "html.parser")
    eras = soup.find_all("table", class_="wikitable sortable")

    images = []
    for table in eras:
        img_els = table.find_all("img")
        images = images + [img.attrs["src"] for img in img_els]

    with progressbar.ProgressBar(max_value=len(images)) as bar:
        for i, img in enumerate(images):
            name = img[img.rindex('/') + 1:]
            name = name.replace("%", "_")
            urllib.request.urlretrieve(
                f"https:{ img }", f"{ outdir }/{ name }")
            bar.update(i)


def resize_images(indir, shape):
    images = listdir(indir)
    with progressbar.ProgressBar(max_value=len(images)) as bar:
        for i, url in enumerate(images):
            image = Image.open(join(indir, url))
            image = image.resize(shape)
            image.save(join(indir, url))
            bar.update(i)


if __name__ == "__main__":
    print("Downloading images...")
    download_images("res")

    print("Resizing images...")
    resize_images("res", (150, 150))
