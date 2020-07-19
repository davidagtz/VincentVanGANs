from argparse import ArgumentParser
from os import listdir, remove
from os.path import join, isfile

parser = ArgumentParser(description="Will delete certain numbers in folder")
parser.add_argument("--inputs", "-I", nargs="+", required=True)
parser.add_argument("--expr", "-E", required=True)
parser.add_argument("--start", type=int)
parser.add_argument("--end", type=int)
parser.add_argument("every", type=int, help="which ones to save")
args = parser.parse_args()


FOLDERS = args.inputs
EVERY = args.every
EXPR = args.expr
FROM = args.start
TO = args.end

if EXPR.index("*") != EXPR.rindex("*"):
    raise Exception("Cannot have multiple asterisks in expression")

first = EXPR[:EXPR.index("*")]
last = EXPR[EXPR.index("*") + 1:]
def get_value(x): return x[x.index(first) + len(first): x.rindex(last)]
def matches(x): return x.startswith(first) and x.endswith(last)


for folder in FOLDERS:
    files = [x for x in listdir(folder) if isfile(join(folder, x))]
    files.sort()
    for file in files:
        if matches(file):
            num = int(get_value(file))

            if FROM is not None and num <= FROM:
                continue
            if TO is not None and num >= TO:
                continue

            start = FROM if FROM is not None else 0
            if (num - start) % EVERY != 0:
                remove(join(folder, file))
