# Heatmaps 

## Methodology
The heatmaps were created by adding the data on the century in which the annotated context was composed to the annotated materials, sorting the resulting dataframe by century from the earliest to the latest, and creating a heatmap with `seabornInstance.heatmap`. Each row representes a context (a sentence or paragraph containing the word), but the contexts themselves are not visible on the heatmap.

## Example 

This heatmap shows the annotation data for the word _salus_, which has three senses. The dark blue marks the highest relevance of the sense to the context, according to the annotator, while the lighter shades mean less relevance and no relevance.[^1]  As the contexts are arranged from earlier to later, one can see that the third sense ('salvation, deliverance from sin') has only been annotated as relevant for contexts starting from 3rd century CE. There has also been more uncertainty in the annotations related to the first sense ('health, welfare') in contexts belonging to later centuries.   

<img src="https://user-images.githubusercontent.com/22611808/166267928-5d0ec14f-bb96-4c6f-88d8-908816e6f0e5.png" width=700>

[^1]: A white cell ('0') means the annotator could not make any decision at all. This did not occur in this example.]
