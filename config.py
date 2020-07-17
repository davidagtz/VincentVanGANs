from configparser import ConfigParser
from os.path import join

CONFIG_NAME = "gans.config"


def config_write(outdir, **kwargs):
    out = ConfigParser()

    default = {}

    for key in kwargs:
        default[key] = str(kwargs[key])

    out["DEFAULT"] = default

    with open(join(outdir, CONFIG_NAME), "w") as config:
        out.write(config)


def config_read(indir, **kwargs):
    read = ConfigParser()
    read.read(join(indir, CONFIG_NAME))

    cf = read["DEFAULT"]
    for key in kwargs:
        if cf.get(key) is None:
            cf[key] = str(kwargs[key])

    return cf
