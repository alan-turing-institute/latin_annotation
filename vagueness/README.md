# Vagueness score

## Contents of this folder

### Folders
- `annotations`: TSV file with the annotation of each lemma. The header contains
  the derivation pattern. The first column shows the era of each annotation context (BC/CE).
- `charts`: plots in PNG format

### Code

- `vagueness.py` is the code that generated most of the results
- `vagueness_chron.py` gets the metadata referred to the periods
- `central_tendencies.py` calculates average, mean, etc.
- `line_plot*.py`, `scatter_plot*.py`, `bar_plot*.py` individual python scripts to generate the plots
- `charts.ipnyb` notebook to explore the plots in a jupyter environment

### Results

- `.tsv` the results with the scores (inputs to generate the plots)
