from collections import Counter
import os
import sys

indir = 'annotations'
odir = 'counts'

if not os.path.exists(odir):
    os.makedirs(odir)

for fn in os.listdir(indir):
    file_annots = []
    ffn = os.path.join(indir, fn)
    ofn = os.path.join(odir, fn)
    lines = [ll.strip() for ll in open(ffn)][1:]
    nbcol = len(lines[0].split(","))
    word = os.path.splitext(fn)[0]
    for line in lines:
        sl = line.split(",")
        if len(sl) != nbcol:
            print(f"bad format: {fn} | {line.strip()}")
            sys.exit(2)
        file_annots.extend(sl)
        file_counts = Counter(file_annots)
        with open(ofn, mode='w') as ofi:
            for ke, va in sorted(file_counts.items()):
                ofi.write(f"{ke},{va}\n")