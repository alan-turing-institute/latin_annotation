# Plots in this folder

All plots in this folder are based on the _confidence score_ of a context, which is defined as the proportion of “4” ratings out of all non-zero and non-one values, expressed as a number in the range from 0 to 1. This value reflects the level of uncertainty involved in the annotation of a given context.

<img src="https://render.githubusercontent.com/render/math?math=confidence=\frac{n_4}{n_2 %2B n_3 %2B n_4}">

## Visualisation of the annotation data

Folder: [all contexts visualisation](https://github.com/alan-turing-institute/latin_annotation/tree/master/words%20visualised/all%20contexts%20visualisation)

Plots in this folder are a visual representation of all annotated contexts from the annotation tasks, in the order in which they were presented to the annotator. Each context from 1 to 60 has a confidence score between 0 and 1, which is plotted. A horizontal line shows the average confidence of annotation for the given word. 

Example: `credo_all_contexts_visualised.png`. The average confidence is 0.6. Most contexts are annotated with a confidence of either 1 or 0.5, apart from a few outliers.
<img src="https://github.com/alan-turing-institute/latin_annotation/blob/ddbb10ed48d72090e02f46972fbbffb19962bbc8/words%20visualised/all%20contexts%20visualisation/credo_all_contexts_visualised.png" width=1000>

## Diachronic visualiation

Folder: [diachronic plots](https://github.com/alan-turing-institute/latin_annotation/tree/master/words%20visualised/diachronic%20plots)

This folder contains the visualisation of the coexistence of different senses of a word, sorted by century. It contains two types of plots, stacked and unstacked. The unstacked plots render the exact number of instances in which a sense occurred in the annotation data. The stacked plots normalise the same data and show the proportion in which  a sense occurs in the same timeframe. 

Example: annotation of _fidelis_. 
1. `fidelis-diachronic-plot-unstacked.png`.
<img src="https://github.com/alan-turing-institute/latin_annotation/blob/ddbb10ed48d72090e02f46972fbbffb19962bbc8/words%20visualised/diachronic%20plots/fidelis-diachronic-plot-unstacked.png" width=1000>

2. `fidelis-diachronic-plot.png`
<img src="https://github.com/alan-turing-institute/latin_annotation/blob/ddbb10ed48d72090e02f46972fbbffb19962bbc8/words%20visualised/diachronic%20plots/fidelis-diachronic-plot.png" width=1000>
