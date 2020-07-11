import requests
import urllib.request
import progressbar
from bs4 import BeautifulSoup


def download_images():
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
            urllib.request.urlretrieve(
                f"https:{ img }", f"res/real/{ img[img.rindex('/') + 1:] }")
            bar.update(i)


if __name__ == "__main__":
    download_images()
