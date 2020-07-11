import requests
import urllib.request
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

    for img in images:
        urllib.request.urlretrieve(
            f"https:{ img }", f"res/{ img[img.rindex('/') + 1:] }")


if __name__ == "__main__":
    download_images()
