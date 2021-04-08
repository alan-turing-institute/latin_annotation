# Vagueness score

## Contents of this folder

### Folders
- `annotations_meta`: TSV file with the annotation of each lemma. The header contains
  the derivation pattern. The first column shows the era of each annotation context (BC/CE).
- `counts`: ancillary folder created with `vagueness.py` that has the number of annotations type (0, 1, 2, 3, 4) by word.
- `charts`: plots in PNG format

### Code

- `vagueness.py` is the code that generated most of the results
- `vagueness_chron.py` gets the metadata referred to the periods
- `central_tendencies.py` calculates average, mean, etc.
- `charts.ipnyb` notebook that generates the plots

### Results

- `.tsv` the results with the scores (inputs to generate the plots)
