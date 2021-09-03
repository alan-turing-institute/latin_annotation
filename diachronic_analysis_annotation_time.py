## -*- coding: utf-8 -*-
# Author: Barbara McGillivray
# Date: 9/6/2021
# Python version: 3
# Script for analysing semantic change from Latin SemEval annotated data to answer the following questions:
# If a sense is new, does it tend to occur in later texts? We do a correlation analysis to see if there's a correlation betweeen century and annotators' ratings of a sense.


# ----------------------------
# Initialization
# ----------------------------


# Import modules:

import os
import csv
import datetime
import re
from collections import Counter
import locale
from pandas import read_excel
import pandas as pd
import xlrd
import numpy as np
import math
from statistics import mean
from scipy.stats import spearmanr
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
#from matplotlib import pyplot
from statistics import mean
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from collections import defaultdict, OrderedDict


# Parameters:

istest_default = "yes"
istest = input("Is this a test? Leave empty for default (" + istest_default + ").")
plot_type = input("Which plot are you interested in? Histogram (hist), line plot for each word (each) or line plot for all words (all)? ")
number_test = 1 # number of words considered when testing

if istest == "":
    istest = istest_default


# Directory and file names:

directory = os.path.join("/Users", "bmcgillivray", "Documents", "OneDrive", "The Alan Turing Institute", "OneDrive - The Alan Turing Institute", "Research", "2019", "Latin corpus")
dir_annotation = os.path.join(directory, "Semantic annotation", "annotated data")
dir_output = os.path.join("/Users", "bmcgillivray", "Documents", "OneDrive", "The Alan Turing Institute", "OneDrive - The Alan Turing Institute", "Research", "2021", "Latin annotation", "Semantic_change_analysis")

# Output file:
output_file_name = "Semantic_change_analysis.csv"

if istest == "yes":
    output_file_name = output_file_name.replace(".csv", "_test.csv")



# This function normalizes the annotators' ratings, because sometimes they marked a number (e.g. "1")
#  and sometimes a string (e.g. "1: Identical":

def normalize_ratings(ratings):
    #print("shape:", str(ratings.shape))
    new_ratings = pd.DataFrame(0, index=np.arange(len(ratings)), columns=range(5, ratings.shape[1] - 1))
    #if len(str(ratings.iloc[1,3])) > 1:
    for column in range(5,ratings.shape[1]-1):
        #print("column", str(column))
        #print("old:\n", str(ratings.iloc[column]))
        #ratings.iloc[:,column] = ratings.iloc[:,column].str.split(': ').str[0]
        new_ratings[column] = [str(ratings.iloc[i, column])[0] for i in range(ratings.shape[0])]
        #print("new:", str(ratings.iloc[:,column]))

    #print("new shape", str(new_ratings.shape))
    #print(str(new_ratings))
    return new_ratings


# output file:
output = open(os.path.join(dir_output, output_file_name), 'w')
output_writer = csv.writer(output, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)


# Read annotation files:

annotator2ratings = dict() # maps an annotator to the data frame of their ratings

words = read_excel(os.path.join(directory, "Semantic annotation", "Target words", "Target_control_words.xlsx"),
                   header=0, index_col=None, names=None, keep_default_na=True)

word_list = words.Word.tolist()
targets = words.loc[words['Type'] == "Target"].Word.tolist()
controls = words.loc[words['Type'] == "Control"].Word.tolist()


if istest == "yes":
    targets = targets[:number_test]
    controls = controls[:number_test]

# I define a dictionary that maps a target word to the list of the indexes of its new senses:
target2newsenses = dict()
for target in targets:
    target2newsenses[target] = words.loc[words['Word'] == target].New_sense.tolist()


# Function that reads the annotated data for a word:
def read_annotation(word):
    file = ""
    for f in os.listdir(dir_annotation):
        if word in f and f.endswith("_metadata.xlsx"):
            file = f

    #print("Reading file", os.path.join(dir_annotation, file))
    ann = read_excel(os.path.join(dir_annotation, file), sheet_name="Annotation", header = 0)
    #print(str(ann.shape[0]), "rows", ann.shape[1], "columns")

    #print(str(ann))
    #print("select columns:")
    #list_columns = [1] + list(range(5, ann.shape[1] - 1))
    #ratings = ann.iloc[0:61, list_columns]
    ratings = normalize_ratings(ann)
    #print(str(ratings))

    eras = ann.iloc[:, 1]
    metadata = ann.iloc[:, 0]
    #print(str(type(metadata)))
    century_list = list()
    metadata_l = metadata.tolist()

    for i in range(len(metadata_l)):

        metadata_f = metadata_l[i]
        #print("\t"+metadata_f)
        metadata_fields = metadata_f.split(",")
        index_of_century = 0

        for j in range(len(metadata_fields)):
            if "cent" in metadata_fields[j]:
                index_of_century = j

        century = metadata_fields[index_of_century]
        century = century.replace(" ", "")
        #print("centuries:",str(century))
        century = century.replace("cent.", "")
        century_sign = ""
        if "B.C." in century and "A.D." in century:
            century_sign = "+"
            century = century.replace("B.C.", "").replace("A.D.", "")
            if "-" in century:
                centuries = century.split("-")
                #print(str(centuries))
                century = 0
        else:
            if "B.C." in century:
                century_sign = "-"
                century = century.replace("B.C.", "")
            elif "A.D." in century:
                century_sign = "+"
                century = century.replace("A.D.", "")
            if "-" in century:
                centuries = century.split("-")
                #print(str(centuries))
                century = mean([int(i) for i in centuries])


        century = float(century)
        if century_sign == "-":
            century = -century
            #print("Negative:", century)
        #print("\tFinal century:", century)

        century_list.append(century)


    return (ratings, eras, metadata, century_list)

# Write header of output file:

if plot_type == "all":
    fig = plt.figure(figsize=(10, 10))

output_writer.writerow(['Target', 'average rating of new senses minus average rating of old senses', 'Spearman correlation coefficient', 'Spearman p value', 'Kendall tau', "Kendall p value"])

colours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'grey', 'pink', 'purple']
count_words = 0
# Read annotated data:
for word in targets:
    count_words += 1
    print("word:", word)
    (ratings_word, eras_word, metadata_word, century_list) = read_annotation(word)
    #print("ratings:",str(ratings_word))
    #print("eras:", str(eras_word))

    # retrieve which sense(s) are/is the new one(s):
    #print(str(target2newsenses[target]))
    #print(str(type(target2newsenses[target])))
    new_senses = target2newsenses[word]
    #print("new senses:",str(new_senses))
    #ratings for new sense(s):
    new_senses = str(new_senses[0]).split(",")

    # find ratings of all old senses:
    #print("number of senses:", str(ratings_word.shape[1]))

    #print("century:", century_list)
    average_new_senses_list = list()
    average_old_senses_list = list()

    for row in range(ratings_word.shape[0]):
        #print("row:", str(row))
        average_new_senses_list_this_row = list()
        average_old_senses_list_this_row = list()

        for sense in range(ratings_word.shape[1]):
            #print("sense number", str(sense))
            if str(sense+1) not in new_senses:
                #print("old sense because", str(sense+1), "is not in ", str(new_senses))
                #print(str(ratings_word.iloc[row, sense]))
                #if eras_word[row].startswith("BC"):
                average_old_senses_list_this_row.append(int(ratings_word.iloc[row, sense]))
            else:
                average_new_senses_list_this_row.append(int(ratings_word.iloc[row, sense]))

        average_new_senses_this_row = mean(average_new_senses_list_this_row)
        average_old_senses_this_row = mean(average_old_senses_list_this_row)
        average_new_senses_list.append(average_new_senses_this_row)
        average_old_senses_list.append(average_old_senses_this_row)

    #print("all_ratings_average_new_senses_list:"+str(average_new_senses_list))
    #print("all_ratings_average_old_senses_list:"+str(average_old_senses_list))

    difference_average_new_old_senses = np.array(average_new_senses_list)-np.array(average_old_senses_list)
    #difference_average_new_old_senses.replace(0, np.nan, inplace=True)
    #difference_average_new_old_senses = difference_average_new_old_senses.astype(np.float).astype("Int32")
    try:
        difference_average_new_old_senses[difference_average_new_old_senses == 0] = np.nan
    except:
        #difference_average_new_old_senses[difference_average_new_old_senses == 0.0] = np.nan
        difference_average_new_old_senses1 = list(difference_average_new_old_senses)
        for i in range(len(difference_average_new_old_senses1)):
            if difference_average_new_old_senses1[i] == 0:
                difference_average_new_old_senses1[i] = np.nan
        difference_average_new_old_senses = np.array(difference_average_new_old_senses1)
        #print(str(type(difference_average_new_old_senses)))
        #print(difference_average_new_old_senses)

    #print("difference:", difference_average_new_old_senses)

    # I calculate the correlation between the list of centuries (time) and the difference betweeen the average ratings of new senses and the average ratings of old senses
    # I remove the NAs:
    century_list_nonas = list()
    difference_average_new_old_senses_nonas = list()
    for i in range(len(century_list)):
        if np.isnan(difference_average_new_old_senses[i]) == False:
            century_list_nonas.append(century_list[i])
            difference_average_new_old_senses_nonas.append(difference_average_new_old_senses[i])

    rho, pval = spearmanr(century_list_nonas, difference_average_new_old_senses_nonas)
    #print("rho, pval", str(rho), str(pval))

    # Kendall tau:
    tau, p_value = stats.kendalltau(century_list_nonas, difference_average_new_old_senses_nonas)


    if plot_type == "all":
        # Line plot for all words:
        d = {'Word' : word, 'Century': century_list_nonas, 'Difference': difference_average_new_old_senses_nonas}
        df = pd.DataFrame(d)
        df = df.sort_values(by=['Century'])
        # I take the average difference for each century:
        df_avg = df.groupby(['Century']).mean()
        df_avg['Century'] = df_avg.index
        df_avg['Word'] = word
        #print(df_avg)
        plt.xlabel('Century')
        plt.ylabel('Difference')
        type = '-b'
        print("count_words", str(count_words))

        if count_words > 10:
            type = '--r'
            colour = colours[count_words-1-10]
        else:
            colour = colours[count_words-1]
        plt.plot(df_avg['Century'], df_avg['Difference'], type, label = word, color = colour)

    # histograms:
    elif plot_type == "hist":
        fig = plt.figure(figsize=(10, 10))
        plt.hist(difference_average_new_old_senses_nonas)
        plt.title('Difference of the two distributions')
        plt.xlabel("value")
        plt.ylabel("Frequency")
        plt.savefig(
            os.path.join(dir_output, "Histogram_distributions_difference_new_old_senses_ratings_" + word + ".png"))
        plt.close(fig)

    else:
        # Line plot for each word:
        fig = plt.figure(figsize=(9, 9))
        d = {'Word': word, 'Century': century_list_nonas, 'Difference': difference_average_new_old_senses_nonas}
        df = pd.DataFrame(d)
        df = df.sort_values(by=['Century'])
        # I take the average difference for each century:
        df_avg = df.groupby(['Century']).mean()
        df_avg['Century'] = df_avg.index
        df_avg['Word'] = word
        #print(df_avg)
        type = '-b'
        print("count_words", str(count_words))
        #plt.title("Distribution of difference for " + word)

        #plt.plot(df_avg['Century'], df_avg['Difference'], type)
        plt.scatter(df_avg['Century'], df_avg['Difference'])
        plt.xlabel("Century")
        plt.ylabel("Difference")
        #ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # main axes
        #ax.plot(df_avg['Century'], df_avg['Difference'], type)
        #ax.set_xlabel('Century', fontsize=5)
        #ax.set_ylabel('Difference', fontsize=5)

        #ax.tick_params(
        #    axis='both',  # changes apply to both axes
        #    which='both',  # both major and minor ticks are affected
        #    bottom=False,  # ticks along the bottom edge are off
        #    left=False,  # ticks along the top edge are off
        #    labelbottom=False,
        #    labelleft=False,
        #    grid_color='grey',
        #    labelsize=5
        #)  # labels along the bottom edge are off
        #ax.set_xticks(np.arange(min(df_avg['Century']), max(df_avg['Century']), step=0.5))
        #ax.set_yticks(np.arange(min(df_avg['Difference']), max(df_avg['Difference']), step=0.1))

        plt.savefig(os.path.join(dir_output, "Plot_distributions_difference_new_old_senses_ratings_" + word + ".png"), bbox_inches='tight')
        plt.close(fig)

    # write to output file:
    output_writer.writerow(
        [word, mean(difference_average_new_old_senses_nonas), rho, pval, tau, p_value])

output.close()

# plot all differences over time for each word:

if plot_type == "all":

    #plt.title('Century vs Difference in new/old sense ratings')
    #plt.xlabel("Century")
    #plt.ylabel("Difference")
    #plt.scatter(century_list_nonas, difference_average_new_old_senses_nonas)
    plt.legend(loc=2, prop={'size': 6}, bbox_to_anchor=(1.05, 1))#, handles = colours[[0]])
    plt.savefig(os.path.join(dir_output, "Plot_distributions_difference_new_old_senses_ratings_allwords.png"), bbox_inches='tight')
    plt.close(fig)