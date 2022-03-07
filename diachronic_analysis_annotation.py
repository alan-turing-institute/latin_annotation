## -*- coding: utf-8 -*-
# Author: Barbara McGillivray
# Date: 9/6/2021
# Python version: 3
# Script for analysing semantic change from Latin SemEval annotated data to answer the following questions:
# If a sense is new, does it tend to occur in later texts?


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
import matplotlib.pyplot as pl
from scipy.stats import wilcoxon

now = datetime.datetime.now()
today_date = str(now)[:10]

# Parameters:

istest_default = "yes"
istest = input("Is this a test? Leave empty for default (" + istest_default + ").")
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

    ann = read_excel(os.path.join(dir_annotation, file), sheet_name="Annotation", header = 0)
    #print(str(ann.shape[0]), "rows", ann.shape[1], "columns")

    #print(str(ann))
    #print("select columns:")
    #list_columns = [1] + list(range(5, ann.shape[1] - 1))
    #ratings = ann.iloc[0:61, list_columns]
    ratings = normalize_ratings(ann)
    #print(str(ratings))

    eras = ann.iloc[:, 1]
    #print("eras:",str(eras))

    return (ratings, eras)

# Write header of output file:

output_writer.writerow(['Target', 'average new senses AD', 'average old senses BC', 'average old senses AD',
                        'average new senses BC', 'average new senses AD + old senses BC', 'average old senses AD + new senses BC',
                        'average new senses AD + average old senses BC > average old senses AD + average new senses BC?', 'difference',
                        "Shapiro test", "p-value of Shapiro test", "Is the distribution normal?", "t-test statistic",
                        "t-test p-value", "Is new senses AD + old senses BC significantly greater than old senses AD + new senses BC? (t-test)", "statistic of Wilcoxon test", "p-value of Wilcoxon test",
                        "Is new senses AD + old senses BC significantly greater than old senses AD + new senses BC? (Wilcoxon test)" ])


# Read annotated data:
for target in targets:
    print(target)
    (ratings_word, eras_word) = read_annotation(target)
    #print("ratings:",str(ratings_word))
    #print("eras:", str(eras_word))

    # calculate average ratings for new senses in BC:
    av_news_bc = 0
    all_ratings_new_senses_bc = list()

    # calculate average ratings for new senses in AD:
    av_news_ad = 0
    all_ratings_new_senses_ad = list()

    # calculate average ratings for old senses in BC:
    av_olds_bc = 0
    all_ratings_old_senses_bc = list()

    # calculate average ratings for old senses in AD:
    av_olds_ad = 0
    all_ratings_old_senses_ad = list()

    # retrieve which sense(s) are/is the new one(s):
    #print(str(target2newsenses[target]))
    #print(str(type(target2newsenses[target])))
    new_senses = target2newsenses[target]
    #print("new senses:",str(new_senses))
    #ratings for new sense(s):
    new_senses = str(new_senses[0]).split(",")

    # find ratings of all new senses:
    #for new_sense in new_senses:
    #    #print(target, "new sense:",str(new_sense))
    #    new_sense = int(new_sense)
    #    #print(str(ratings_word.iloc[0:61, new_sense - 1]))
    #    average_new_senses_row_bc = list()
    #    average_new_senses_row_ad = list()
    #    for row in range(ratings_word.shape[0]):
    #        #print("row:", str(row))
    #        #print(str(ratings_word.iloc[row, :]))
    #        ##print(str(ratings_word.iloc[row,new_sense-1]))
    #        #print("eras_word:", str(eras_word[row]))
    #        if eras_word[row].startswith("BC"):
    #            average_new_senses_row_bc.append(int(ratings_word.iloc[row,new_sense-1]))
    #        else:
    #            average_new_senses_row_ad.append(int(ratings_word.iloc[row, new_sense - 1]))



    # find ratings of all old senses:
    #print("number of senses:", str(ratings_word.shape[1]))
    for row in range(ratings_word.shape[0]):
        print("row:", str(row))
        average_old_senses_row_bc = list()
        average_old_senses_row_ad = list()
        average_new_senses_row_bc = list()
        average_new_senses_row_ad = list()
        for sense in range(ratings_word.shape[1]):
            #print("sense number", str(sense))
            if str(sense+1) not in new_senses:
                #print("old sense because", str(sense+1), "is not in ", str(new_senses))
                #print(str(ratings_word.iloc[row, sense]))
                if eras_word[row].startswith("BC"):
                    average_old_senses_row_bc.append(int(ratings_word.iloc[row, sense]))
                    print("average_old_senses_row_bc:", str(average_old_senses_row_bc))
                else:
                    average_old_senses_row_ad.append(int(ratings_word.iloc[row, sense]))
                    print("average_old_senses_row_ad:",str(average_old_senses_row_ad))
            else:
                if eras_word[row].startswith("BC"):
                    average_new_senses_row_bc.append(int(ratings_word.iloc[row, sense]))
                    print("average_new_senses_row_bc:",str(average_new_senses_row_bc))
                else:
                    average_new_senses_row_ad.append(int(ratings_word.iloc[row, sense]))
                    print("average_new_senses_row_ad:",str(average_new_senses_row_ad))

        try:
            all_ratings_old_senses_bc.append(mean(average_old_senses_row_bc))
            print("mean(average_old_senses_row_bc):",str(mean(average_old_senses_row_bc)))
        except:
            print("no items in all_ratings_old_senses_bc")
        try:
            all_ratings_old_senses_ad.append(mean(average_old_senses_row_ad))
            print("mean(average_old_senses_row_ad):", str(mean(average_old_senses_row_ad)))
        except:
            print("no items in all_ratings_old_senses_ad")
        try:
            all_ratings_new_senses_bc.append(mean(average_new_senses_row_bc))
            print("mean(average_onew_senses_row_bc):", str(mean(average_new_senses_row_bc)))
        except:
            print("no items in all_ratings_new_senses_bc")
        try:
            all_ratings_new_senses_ad.append(mean(average_new_senses_row_ad))
            print("mean(average_new_senses_row_ad):", str(mean(average_new_senses_row_ad)))
        except:
            print("no items in all_ratings_new_senses_ad")

    print("all_ratings_new_senses_bc:", (all_ratings_new_senses_bc))
    print("all_ratings_new_senses_ad:", (all_ratings_new_senses_ad))
    print("all_ratings_old_senses_bc:", (all_ratings_old_senses_bc))
    print("all_ratings_old_senses_ad:", (all_ratings_old_senses_ad))
    #print("sum:", str(sum(all_ratings_new_senses_bc)))
    #print("sh:", str(ratings_word.shape[0]))
    #print("den:", str(ratings_word.shape[0]*len(new_senses)))
    av_news_bc = float(sum(all_ratings_new_senses_bc)/ratings_word.shape[0]*len(new_senses))
    #print("av_news_bc:",str(av_news_bc))
    av_olds_bc = float(sum(all_ratings_old_senses_bc) / (ratings_word.shape[0] * (ratings_word.shape[1]-len(new_senses))))
    #print(str(ratings_word.shape[0] * (ratings_word.shape[1]-len(new_senses))))
    #print("av_olds_bc:", str(av_olds_bc))
    av_news_ad = float(sum(all_ratings_new_senses_ad) / ratings_word.shape[0] * len(new_senses))
    #print("av_news_ad:", str(av_news_ad))
    av_olds_ad = float(
        sum(all_ratings_old_senses_ad) / (ratings_word.shape[0] * (ratings_word.shape[1] - len(new_senses))))
    # print(str(ratings_word.shape[0] * (ratings_word.shape[1]-len(new_senses))))
    #print("av_olds_ad:", str(av_olds_ad))
    hypothesis_true = ""
    if av_news_ad+av_olds_bc>av_olds_ad+av_news_bc:
        hypothesis_true = "yes"
    else:
        hypothesis_true = "no"

    # Hypothesis: newAD+oldBC>oldAD+newBC:
    #output.write(target+"\taverage new senses AD:"+ str(av_news_ad)+ "\t"+ "average old senses BC:"+ str(av_olds_bc)+ "\t"+
    #             "average old senses AD:"+ str(av_olds_ad)+ "\t"+ "average new senses BC:"+ str(av_news_bc)+ "\tnew senses AD + old senses BC:"+
    #             str(av_news_ad+av_olds_bc)+ "\t"+ "old senses AD + new senses BC:"+ str(av_olds_ad+av_news_bc)+ "\t"+
    #             "new senses AD + old senses BC > old senses AD + new senses BC? "+ "\t"+ hypothesis_true+"difference:"+str((av_news_ad+av_olds_bc)-(av_olds_ad+av_news_bc))+"\t")

    # Paired one-tailed t-test to test whether the mean of all_ratings_new_senses_ad+all_ratings_old_senses_bc is significantly greater than all_ratings_old_senses_ad+all_ratings_new_senses_bc
    # Assumptions:
    # 1) Differences between the two dependent variables follows an approximately normal distribution (Shapiro-Wilks Test);
    # 2) Independent variable should have a pair of dependent variables; ???
    # 3) Differences between the two dependent variables should not have outliers YES
    # 4) Observations are sampled independently from each other YES
    # Null hypothesis: the distribution of all_ratings_new_senses_ad+all_ratings_old_senses_bc is the same as all_ratings_old_senses_ad+all_ratings_new_senses_bc

    print("all_ratings_new_senses_ad+all_ratings_old_senses_bc:"+str(all_ratings_new_senses_ad+all_ratings_old_senses_bc))
    print("all_ratings_old_senses_ad+all_ratings_new_senses_bc:"+str(all_ratings_old_senses_ad+all_ratings_new_senses_bc))
    a = all_ratings_new_senses_ad+all_ratings_old_senses_bc
    b = all_ratings_old_senses_ad+all_ratings_new_senses_bc
    #a = [math.log(x+0.00000001) for x in a]
    #b = [math.log(x+0.00000001) for x in b]
    x = np.array(a)-np.array(b)
    #print(str(x))
    # Shapiro-Wilks test to test that the difference between the two distributions is normally distributed:
    shapiro_test = stats.shapiro(x)
    #output.write("Shapiro test:" + str(shapiro_test.statistic) + ", p-value:" + str(shapiro_test.pvalue)+"\t")
    alpha = 0.05
    is_normal = ""
    if shapiro_test.pvalue > alpha:
        is_normal = "yes"
    else:
        is_normal = "no"
    # print histogram:
    fig = pl.hist(x)
    pl.title('Difference of the two distributions')
    pl.xlabel("value")
    pl.ylabel("Frequency")
    pl.savefig(os.path.join(dir_output, "Histogram_distributions_"+target+".png"))
    # Quantile-quantile plot:
    fig2 = qqplot(x, line='s')
    pl.title('Difference of the two distributions')
    pl.xlabel("value")
    pl.ylabel("Frequency")
    pl.savefig(os.path.join(dir_output, "QQplot_distributions_"+target+".png"))

    # t test:
    t, p = stats.ttest_ind(a, b)
    is_greater = ""
    if p > alpha:
        is_greater = "no"
    else:
        is_greater = "yes"
    #print("t = " + str(t))
    #print("p = " + str(p))
    #output.write("t = " + str(t), "p = " + str(p))
    # NB: the t-test can't be used because the distributions aren't normal.

    # Wilcoxon signed-rank test:
    # The assumption made for the Wilcoxon test is that the variable being tested is symmetrically distributed about the median, which would also be the mean.
    # Remember too that it is still vitally important that your sample has been randomly chosen from the population.
    # Data are paired and come from the same population.
    # Each pair is chosen randomly and independently[
    # The data are measured on at least an interval scale; if does suffice that within-pair comparison are on an ordinal scale
    w, p = wilcoxon(a,b, alternative='greater')
    is_greater_w = ""
    if p > alpha:
        is_greater_w = "no"
    else:
        is_greater_w = "yes"

    # write to output file:
    output_writer.writerow(
        [target, av_news_ad, av_olds_bc, av_olds_ad, av_news_bc, av_news_ad + av_olds_bc, av_olds_ad + av_news_bc,
         hypothesis_true, (av_news_ad + av_olds_bc) - (av_olds_ad + av_news_bc), shapiro_test.statistic, shapiro_test.pvalue, is_normal, t, p, is_greater, w, p, is_greater_w])

output.close()