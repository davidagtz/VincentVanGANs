from configparser import ConfigParser
from os import listdir
from os.path import join, isfile, isdir

CONFIG_NAME = "gans.config"


def config_write(outdir, indirs=None, **kwargs):
    out = ConfigParser()

    out["DEFAULT"] = {}

    for key in kwargs:
        if key == "PATHS":
            raise Exception("Cannot use PATHS as variable name")
        out["DEFAULT"][key] = str(kwargs[key])

    if indirs is not None:
        paths = {}
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
        values["PATHS"] = paths
    else:
        values["PATHS"] = indirs

    return values


def get_number(exp, folder, isdir=False):
    if exp.index("*") != exp.rindex("*"):
        raise Exception("String can only have one asterisk")

    files = []
    if isdir:
        [x for x in listdir(folder) if isdir(x)]
    else:
        [x for x in listdir(folder) if isfile(x)]

    FIRST = exp[:exp.index("*")]
    SECOND = exp[exp.rindex("*") + 1:]
    max_n = 0

    for file in files:
        if file.startswith(FIRST) and file.endswith(SECOND):
            num = int(file[len(FIRST): len(file) - len(SECOND)])
            max_n = max(num, max_n)

    return max_n
