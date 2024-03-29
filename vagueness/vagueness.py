from collections import Counter
import os
import sys
import statistics

indir = 'annotations'
out_word = 'ratios_by_word.tsv'
out_source_nbr = 'ratios_by_source_nbr.tsv'
out_meaning_nbr_ave = 'ratios_by_meaning_nbr_ave.tsv'
out_meaning_nbr_med = 'ratios_by_meaning_nbr_median.tsv'
out_pattern = 'ratios_by_pattern.tsv'


by_source_nbr = {}
by_meaning_nbr = {}
by_derivation_pattern = {}
with open(out_word, mode='w') as out_all:
    # collect annotation counts per word and compute ratios
    # each file has data for one word
    for fn in sorted(os.listdir(indir)):
        file_annots = []
        ffn = os.path.join(indir, fn)
        lines = [ll.strip() for ll in open(ffn)]
        # first column contains metadata related to the era
        nbmeanings = len(lines[0].split(",")) - 1
        # first line contains the derivation pattern
        nblines = len(lines[1:])
        word = os.path.splitext(fn)[0]
        first_line = lines[0]
        first_line_list = first_line.split(",")
        number_of_bases = first_line_list.count('base')
        if number_of_bases == 0:
            print(f"wrong! {fn}")
            sys.exit(2)
        number_of_sources = len(set(first_line_list)) - number_of_bases - 1
        if number_of_sources < 1:
            derivation_pattern = f"{number_of_bases}b"
        else:
            derivation_pattern = f"{number_of_bases}b, {number_of_sources}s"
        for line in lines[1:]:
            sl = line.split(",")
            sl = sl[1:]
            if len(sl) != nbmeanings:
                print(f"bad format: {fn} | {line.strip()}")
                sys.exit(2)
            file_annots.extend([int(ann) for ann in sl])
            file_counts = Counter(file_annots)
        #alpha is the vagueness score
        alpha = (file_counts[3] + file_counts[4]) / (nbmeanings * nblines) - 1 / nbmeanings
        alpha = alpha / (1 - 1 / nbmeanings)
        #print output by word
        out_all.write(f"{word}\t{alpha:.3f}\n")
        #collect values by number of sources
        by_source_nbr.setdefault(number_of_sources, {"alpha": []})
        by_source_nbr[number_of_sources]["alpha"].append(alpha)
        #collect values by number of meanings
        by_meaning_nbr.setdefault(nbmeanings, {"alpha": []})
        by_meaning_nbr[nbmeanings]["alpha"].append(alpha)
        #collect values by derivation pattern
        by_derivation_pattern.setdefault(derivation_pattern, {"alpha": []})
        by_derivation_pattern[derivation_pattern]["alpha"].append(alpha)

with open(out_source_nbr, mode='w') as averages:
    for nbr, vals in sorted(by_source_nbr.items()):
        vals["alpha_ave"] = sum(vals["alpha"])/len(vals["alpha"])
        averages.write(f"{nbr}\t{vals['alpha_ave']:.3f}\n")

with open(out_meaning_nbr_ave, mode='w') as averages:
    for nbr, vals in sorted(by_meaning_nbr.items()):
        vals["alpha_ave"] = sum(vals["alpha"])/len(vals["alpha"])
        averages.write(f"{nbr}\t{vals['alpha_ave']:.3f}\n")

with open(out_meaning_nbr_med, mode='w') as medians:
    for nbr, vals in sorted(by_meaning_nbr.items()):
        vals["alpha_med"] = statistics.median(vals["alpha"])
        medians.write(f"{nbr}\t{vals['alpha_med']:.3f}\n")

with open(out_pattern, mode='w') as averages:
    for pattern, vals in sorted(by_derivation_pattern.items()):
        vals["alpha_ave"] = sum(vals["alpha"])/len(vals["alpha"])
        averages.write(f"{pattern}\t{vals['alpha_ave']:.3f}\n")
