# Latin annotation 
Code for analysing the semantic annotation of Latin data from SemEval 2020 task 1

`interannotator_latinise.py` (developed by Barbara McGillivray): calculates inter-annotator agreement between the annotators that annotated the word virtus.

`diachronic_analysis.py` (developed by Barbara McGillivray): analyses the association between new senses of target words and CE texts.

`confidence_analysis.py` (developed by Daria Kondakova): analyses factors influencing the annotation confidence.

## Confidence analysis

The setup of the annotation task meant that each word was only annotated by one person, apart from _uirtus_. To account for the potential differences between the individual annotators, we conducted a quantitative analysis of the annotated data. The objectives of the analysis were: (1) to find out whether there is a personal style of annotation that would affect further analysis of the data; and (2) to look for features of the words themselves that could influence the annotators’ decisions. 

The results of the first part of the analysis are presented in the form of spreadsheets with data aggregated by (a) annotator and (b) number of senses of the annotated word, stored in [confidence analysis/spreadsheets](https://github.com/alan-turing-institute/latin_annotation/tree/master/confidence%20analysis/spreadsheets). 

Folders [heatmaps](https://github.com/alan-turing-institute/latin_annotation/tree/master/heatmaps) and [words visualised](https://github.com/alan-turing-institute/latin_annotation/tree/master/words%20visualised) contain the visualisation of the annotation data on the level of an individual word. More information on the specific visualisations can be found in the respective folders. 

The commented code can be viewed in the Jupyter notebook `confidence_analysis.ipynb`. 

## Vagueness score
Code to calculate the vagueness score of each word and related plots is contained in the folder [vagueness](https://github.com/alan-turing-institute/latin_annotation/tree/master/vagueness). The folder contains additional documentation.

