from collections import Counter
import os
import sys

indir = 'annotations_meta'
out_word_chron = 'ratios_by_word_chron.tsv'

with open(out_word_chron, mode='w') as out_all:
    # collect annotation counts per word and compute ratios
    # each file has data for one word
    for fn in sorted(os.listdir(indir)):
        file_annots_bc = []
        file_annots_ad = []
        ffn = os.path.join(indir, fn)
        lines = [ll.strip() for ll in open(ffn)]
        nbmeanings = len(lines[0].split(",")) - 1
        data = open(ffn).read()
        nblines_bc = data.count("BC")
        nblines_ad = data.count("AD")
        word = os.path.splitext(fn)[0]
        for line in lines[1:]:
            sl = line.split(",")
            if 'AD' in sl:
                file_annots_ad.extend([int(ann) for ann in sl[1:]])
                file_counts_ad = Counter(file_annots_ad)
            elif 'BC' in sl:
                file_annots_bc.extend([int(ann) for ann in sl[1:]])
                file_counts_bc = Counter(file_annots_bc)
            else:
                print(f"error: {fn}")
        comp_bc = (file_counts_bc[4] + file_counts_bc[3]) / (nbmeanings * nblines_bc) - 1 / nbmeanings
        comp_ad = (file_counts_ad[4] + file_counts_ad[3]) / (nbmeanings * nblines_ad) - 1 / nbmeanings
        # print bc/ad value of “alpha” (competition) by word
        out_all.write(f"{word}\t{comp_bc:.3f}\t{comp_ad:.3f}\n")
