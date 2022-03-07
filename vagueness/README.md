# Vagueness score

* [Methodology](#methodology)
  * [Formula](#formula)
* [Contents of this folder](#contents-of-this-folder)
  * [Folders](#folders)
  * [Code](#code)
  * [Results](#results)

## Methodology

In order to calculate the vagueness of a word we devised the following methodology. In the word’s annotation, an annotation labelled “4” (the word’s usage displays the same meaning as the dictionary sense in question) followed by one or more “1” (the word’s usage is unrelated to the sense) implies that the meaning of the word in the given context was clear, i.e., that there is no vagueness (see Table 1). On the contrary, when there is more than one “4”, more than one “3” or the presence of both “3” and “4”, this means that the meaning of the word in the given context was not clear and that several interpretations were possible, i.e., it is a case of vagueness (see Table 2). The ideal case of absence of vagueness is an annotation in which only one meaning is annotated with a “4” or a “3” and that all the other senses are annotated with a “1” or a “2”. The vagueness score aims to represent any deviation from this ideal case. 

**Table 1**: Example of annotation with no vagueness (lemma: _adsumo_).

| Left context                                                                                                                                                                                                                                                                                                                                                                                                                                        | word        | Right context                                                                                                                                                                                                                                                       | Sense 1: take to oneself | Sense 2: receive |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|------------------|
| Haec dum in India geruntur, Graeci milites nuper in colonias a rege deducti circa Bactra orta inter ipsos seditione defecerant, non tam Alexandro infensi quam metu supplicii. Quippe, occisis quibusdam popularium, qui ualidiores erant, arma spectare coeperunt et Bactriana arce, quae casu neglegentius adseruata erat, occupata Barbaros quoque in societatem defectionis inpulerant. Athenodorus erat princeps eorum, qui regis quoque nomen | adsumpserat | , non tam imperii cupidine quam in patriam reuertendi cum iis, qui auctoritatem ipsius sequebantur. Huic Biton quidam nationis eiusdem, sed ob aemulationem infestus, conparauit insidias, inuitatumque ad epulas per Boxum quendam Bactrianum in conuiuio occidit. | 4: Identical             | 1: Unrelated     |



**Table 2**: Example of annotation with vagueness (lemma: _adsumo_).

| Left context                                                                                                                                                                                                                                                                                                                                                                                                                                        | word        | Right context                                                                                                                                                                                                                                                       | Sense 1: take to oneself | Sense 2: receive |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|------------------|
| Haec dum in India geruntur, Graeci milites nuper in colonias a rege deducti circa Bactra orta inter ipsos seditione defecerant, non tam Alexandro infensi quam metu supplicii. Quippe, occisis quibusdam popularium, qui ualidiores erant, arma spectare coeperunt et Bactriana arce, quae casu neglegentius adseruata erat, occupata Barbaros quoque in societatem defectionis inpulerant. Athenodorus erat princeps eorum, qui regis quoque nomen | adsumpserat | , non tam imperii cupidine quam in patriam reuertendi cum iis, qui auctoritatem ipsius sequebantur. Huic Biton quidam nationis eiusdem, sed ob aemulationem infestus, conparauit insidias, inuitatumque ad epulas per Boxum quendam Bactrianum in conuiuio occidit. | 4: Identical             | 1: Unrelated     |


### Formula 

The vagueness score is calculated as follows (see formula below). We first calculate the sum of the ratings “3” _– count(3) –_ and “4” – _count(4)_ – for each lemma. This sum is then divided by the number of annotated passages (y, usually 60) times the number of meanings (_n_). This gives us the average number of “3’s” and “4’s” per cell in the annotation, as each word’s annotation consists of _n_ x _y_ cells to be assigned a value between “0” and “4”. Considering that, if the meanings are clearly distinct, the absence of vagueness presupposes that only one meaning is annotated with “4” (or “3”), the expected value of this average would be 1 divided by the number of meanings. Consequently, we calculate the deviation from this pattern by subtracting a fraction of 1 over the number of meanings of the word (v in the formula below) from the obtained average. However, this deviation needs to be scaled: while the ideal case of absence of vagueness will always yield 0, the maximum value of _v_ would depend on the number of meanings, hence the required correction. First, we calculate the maximum value of _v_, which is 1 minus 1 divided by the number of meanings (_vmax_ below). Then, the scaled value of the deviation score _v_ (_vscaled_) is equal to dividing _v_ by its possible maximum value. Thanks to this operation, the maximum score of vagueness is always 1 regardless of the number of meanings of the word. For instance, the expected average of 3s and 4s of a two-meaning word with no vagueness is equal to 0.5 (that is, for each passage, one of the meanings was annotated with a “4” and the other with a “1”). In Tables 1 and 2 we present two excerpts of the annotation for the lemma _adsumo_. If these annotations represented all the annotations we had for adsumo, the average of values “3” and “4” would be 0.75. Therefore, if we subtract to this value the expected average if there was no vagueness, we obtain a deviation score _v_ of 0.25. The maximum value of _v_ for a two-sense word is 0.5, thus to calculate the vagueness score of this example of adsumo we divide 0.25 (_v_) by 0.5 (_vmax_) and we obtain a vagueness score of 0.5.

<p align="center"><img align="center" src="https://i.upmath.me/svg/v%20%3D%20%7Bcount(3)%20%2B%20count(4)%20%5Cover%20n%20%5Ctimes%20y%7D%20-%20%7B1%20%5Cover%20n%7D" alt="v = {count(3) + count(4) \over n \times y} - {1 \over n}" /></p>


<p align="center"><img align="center" src="https://i.upmath.me/svg/v_%7Bmax%7D%20%3D%201%20-%20%7B1%20%5Cover%20n%7D" alt="v_{max} = 1 - {1 \over n}" /></p>

<p align="center"><img align="center" src="https://i.upmath.me/svg/v_%7Bscaled%7D%20%3D%20%7Bv%20%5Cover%20v_%7Bmax%7D%7D" alt="v_{scaled} = {v \over v_{max}}" /></p>



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
