import requests
import urllib.request
import progressbar
import numpy as np
import sys
from bs4 import BeautifulSoup
from PIL import Image
from os import listdir, remove, mkdir
from os.path import isfile, join, exists, isdir
from argparse import ArgumentParser
from click import confirm
from shutil import rmtree


def download_images(outdir, **kwargs):
    res = requests.get(
        "https://en.wikipedia.org/wiki/List_of_works_by_Vincent_van_Gogh")

    if int(res.status_code / 100) != 2:
        raise Exception("Error in receiving images")

    res = res.content

    soup = BeautifulSoup(res, "html.parser")
    eras = soup.find_all("table", class_="wikitable sortable")

    images = []
    SIZE = 0
    if kwargs["separate"] != None:
        for table in eras:
            img_els = table.find_all("img")
            urls = [img.attrs["src"] for img in img_els]
            header = table.find_previous_sibling(
                ["h2", "h3"]).find(class_="mw-headline")
            urls.insert(0, header.string)
            images.append(urls)
            SIZE += len(urls) - 1
    else:
        for table in eras:
            img_els = table.find_all("img")
            urls = [img.attrs["src"] for img in img_els]
            images += urls
        SIZE = len(images)

    with progressbar.ProgressBar(max_value=SIZE) as bar:
        if kwargs["separate"] == None:
            for i, img in enumerate(images):
                ext = img[img.rindex('.') + 1:]
                urllib.request.urlretrieve(
                    f"https:{ img }", f"{ outdir }/{ i }.{ ext }")
                bar.update(i)
        else:
            i = 0
            for section in images:
                title = section[0].replace(" ", "_")
                folder = join(outdir, title)
                mkdir(folder)
                for j in range(1, len(section)):
                    img = section[j]
                    ext = img[img.rindex('.') + 1:]
                    urllib.request.urlretrieve(
                        f"https:{ img }", join(folder, f"{ j }.{ ext }"))
                    bar.update(i)
                    i += 1


RESIZE_I = 0


def resize_images(indir, shape):
    global RESIZE_I

    sys.stdout.write(f"\r{ RESIZE_I }")

    images = listdir(indir)
    for url in images:
        actual = join(indir, url)
        if isdir(actual):
            resize_images(actual, shape)
        else:
            image = Image.open(actual)
            image = image.resize(shape)
            image.save(actual)
            RESIZE_I += 1
            sys.stdout.write("\r             ")
            sys.stdout.write(f"\r{ RESIZE_I }")


def refresh_dir(dir, shape, **kwargs):
    if dir.strip() == "":
        raise Exception("No directory given")

    if isfile(dir):
        if not confirm("Overwrite file?"):
            exit()
        else:
            remove(dir)
    elif exists(dir):
        if not confirm("Overwrite directory?"):
            exit()
        else:
            rmtree(dir)

    mkdir(dir)

    print("Downloading images...")
    download_images(dir, **kwargs)

    print("Resizing images...")
    resize_images(dir, shape)
    print("")


if __name__ == "__main__":
    parser = ArgumentParser(description="Downloads and resizes images")

    parser.add_argument("--separate", action="store_true")
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--out", default="assets/res")

    args = parser.parse_args()

    refresh_dir(args.out, (args.size, args.size), separate=args.separate)
