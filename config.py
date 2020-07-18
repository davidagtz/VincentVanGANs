from configparser import ConfigParser
from os.path import join

CONFIG_NAME = "gans.config"


def config_write(outdir, indirs=None, **kwargs):
    print("Wrting config...")
    out = ConfigParser()

    out["DEFAULT"] = {}

    for key in kwargs:
        if key == "PATHS":
            raise Exception("Cannot use PATHS as variable name")
        out["DEFAULT"][key] = str(kwargs[key])

    if indirs is not None:
        paths = {}
        print("HUH ", indirs)
        for i, value in enumerate(indirs):
            paths[str(i)] = value
        out["PATHS"] = paths

    with open(join(outdir, CONFIG_NAME), "w") as config:
        out.write(config)


def config_read(indir, indirs=[], **kwargs):
    values = {}

    read = ConfigParser()
    read.read(join(indir, CONFIG_NAME))

    cf = read["DEFAULT"]
    for key in kwargs:
        if cf.get(key) is None:
            values[key] = kwargs[key]

    for key, value in cf.items():
        values[key] = value

    if "PATHS" in read:
        paths = []
        for key in read["PATHS"]:
            if key not in read["DEFAULT"]:
                paths.append(read["PATHS"][key])
        print(paths)
        values["PATHS"] = paths
    else:
        values["PATHS"] = indirs

    return values
