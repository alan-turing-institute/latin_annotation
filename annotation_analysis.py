#!/usr/bin/env python
# coding: utf-8

# In[ ]:


__author__ = 'dareia'


# In[ ]:


import os
import pandas as pd
import numpy as np
from pandas import read_excel
pd.options.display.max_colwidth = 100
import re
import pprint
pp = pprint.PrettyPrinter()
import matplotlib as mpl
import matplotlib.pyplot as plt
import ast
import itertools
import textwrap
import seaborn as seabornInstance;  seabornInstance.set()
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


mpl.rcParams['figure.dpi']= 500
mpl.rc("savefig", dpi=500)
plt.rcParams['figure.figsize'] = [7, 20]
patterns = ["|", "\\", "/", "+", "-", ".", "*", "x", "o", "O"]


# In[ ]:


from adjustText import adjust_text
def plot_scatter(df, x_values, y_values, labels, filename=None, a=0.4, c='blue', m='o'):
    #plt.clf()
    df.plot.scatter(ax=ax, y=y_values, x=x_values, alpha=a, c=c, marker=m, s=80)
    if labels==None:
        pass
    else:
        texts = []
        for x, y, s in df[[x_values, y_values, labels]].itertuples(index=False, name=None):
            texts.append(plt.text(x, y, s))
        adjust_text(texts) #, x=x_values, y=y_values, autoalign='y',
                #only_move={'points':'y', 'text':'y'}, force_points=0.15,
                #arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    if filename == None:
        filename = 'confidence analysis/{value1}_and_{value2}_scatter.png'.format(value1=y_values, value2=x_values)
    plt.savefig(filename, bbox_inches='tight',
                #transparent=True
               )


# In[ ]:


def printsep(s):
    print(s*30)


# In[ ]:


def normalize_ratings(ratings):
    printsep('+')
    print("Normalizing ratings...")
    print("shape:", str(ratings.shape))
    ratings = ratings.dropna()
    for i in ratings.index:
        for y in range(ratings.shape[1]):
            if len(str(ratings.iat[i, y])) > 3:
                ratings.iat[i, y] = int(ratings.iat[i, y].split(': ')[0])
    print(str(ratings))
    print("... ratings normalized")
    printsep('+')
    return ratings


# In[ ]:


def annotator_probability(maxmeanings, probs):
    ratings = [0, 1, 2, 3, 4]
    combinations = []
    pattern_probability = {}
    for i in range(2,maxmeanings+1):
        combinations.extend([list(p) for p in itertools.combinations_with_replacement(ratings, i)])
    for pattern in combinations:
        probability = 1
        for elem in pattern:
            try:
                probability *= probs[elem]
            except KeyError:            # this handles the cases where an annotator never used a rating
                probability *= 0
        pattern_probability[str(pattern)] = probability
    return pattern_probability


# # Reading data from spreadsheets (.xslx)

# In[ ]:


directory = os.path.join("..", "..", "Annotated data")


# In[ ]:


paths = [f for f in os.listdir(directory) if f[-5:]=='.xlsx' and f[0]!='~']
print(paths)


# In[ ]:


spreadsheets = []
single_words = {}


# In[ ]:


# This cell runs the loop on one file.
for path in paths:
    printsep('-|-|')
    print('Working with the file ' + path)
    s = {} # will write all information from the spreadsheet to this dict
    split = re.split('_', path)
    s['annotator key'] = split[3]
    s['word'] = split[2]
    # OPEN FILE
    df = read_excel(os.path.join(directory, path),
                                  sheet_name=0, encoding='utf-8', dtype=object)
    df.columns = [name.lower() for name in df.columns]
    df.replace('', np.NaN, inplace=True)
    df = df.dropna(how='all')
    print(df.tail())
    #s['file'] = df # not sure i need that
    # GET MEANINGS AND THEIR NUMBER
    column_headings = list(df.columns)
    print(column_headings)
    try:
        x = column_headings.index('right context')+1 # where metadata and test sentence ends
        y = column_headings.index('comments') # the last column (also checks whether it is "comments")
        meanings = [m.replace('\xa0', ' ') for m in column_headings[x:y]] # dealing with the \xa0 character
    except ValueError:
        print("No such columns in the spreadsheet.")
        meanings = []
        break
    print('Extracted meanings are:')
    print(meanings)
    s['meanings'] = meanings
    #add number of meanings to the dictionary
    s['number of meanings'] = len(meanings)
    #print(s['cells'])
    # FIND COMMENTS, COUNT THE NUMBER
    comments = df.dropna()
    print(comments)
    s['comments'] = comments
    s['number of comments'] = comments.shape[0]
    # SELECT A SUBSET WITH ONLY MEANINGS AND RATINGS + INDEX
    df_ratings = df.iloc[:, x:y]
    df_ratings = normalize_ratings(df_ratings)
    s['rows'] = df_ratings.shape[0]
    s['cells'] = s['rows'] * s['number of meanings']
    print(df_ratings.head())
    printsep('-')
    # COUNT UNIQUE PATTERNS OF ANNOTATION
    all_rating_patterns = []
    for i in range(df_ratings.shape[0]):
        pattern = str((sorted(df_ratings.iloc[i].to_list())))
        all_rating_patterns.append(pattern)
    unique_rating_patterns = set(all_rating_patterns)
    rating_patterns_counted = {}
    for p in all_rating_patterns:
        if p in rating_patterns_counted.keys():
            rating_patterns_counted[p] += 1
        else:
            rating_patterns_counted[p] = 1
    #pp.pprint(rating_patterns_counted)
    #print(rating_patterns_counted)
    # turn dictionary into a Series, then DataFrame 
    rating_patterns_counted = pd.Series(rating_patterns_counted)
    rating_patterns_counted = pd.DataFrame({'pattern': rating_patterns_counted.index, 'count': rating_patterns_counted.values})      # result
    print("Unique rating patterns:")
    pp.pprint(rating_patterns_counted)
    s['rating patterns'] = rating_patterns_counted
    printsep('-')
    # COUNT ALL VALUES ACCROSS THE DATAFRAME
    # stack all columns, do value_counts()
    arc = df_ratings.stack().value_counts()
    # add absolute numbers and their relative frequency to a dataframe
    arc = pd.DataFrame({'rating':arc.index,
                                        'absolute number':arc.values,
                                        'frequency': arc.values / s['cells']})   # result
    print("All ratings:")
    print(arc)
    try:
        fours = arc.loc[arc['rating'] == 4, 'absolute number'].item()
    except ValueError:
        fours = 0
    try:
        threes = arc.loc[arc['rating'] == 3, 'absolute number'].item()
    except ValueError:
        threes = 0
    try:
        twos = arc.loc[arc['rating'] == 2, 'absolute number'].item()
    except ValueError:
        twos = 0
    conf =  fours / (fours + threes + twos)
    
    s['individual ratings frequency'] = arc
    s['confidence'] = conf
    s['average'] = (fours * 4 + threes * 3 + twos * 2) / ((fours + threes + twos) * 4)
    s['ratings'] = df_ratings
    # SHOW RESULTING DICTIONARY
    pp.pprint(s)
    # ADD THIS WHOLE THING TO THE LIST OF DATA ON SPREADSHEETS
    spreadsheets.append(s)
    # plot a heatmap of ratings, sorted by date
    metadata = df['metadata']
    dates = []
    for line in metadata:
        #print(line)
        try:
            match = re.search(',cent\. (.+?),', line)
            if match:
                datestr = match.group(1)
                century = datestr.split()[0]
                era = datestr.split()[1]
                #print(datestr)
                if '-' in century:
                    date = (float(century.split('-')[0])+int(century.split('-')[1]))/2
                else:
                    date = float(century)
                if datestr.split()[1] == 'B.':
                    date = float(-date)
                #print(date)
                dates.append(date)
            else: 
                print('ERROR! '*3)
                print(line)
                while True:
                    try:
                        date = float(input('Input century manually. For 3 BC, -3; for 10 AD, 10; for 2-3 AD, 2.5 etc.'))
                    except ValueError:
                        print("Not a valid number! Try again.")
                        continue
                    else:
                        dates.append(date)
                        break 
        except TypeError:
            continue 
    # CREATE A HEATMAP FOR THE CURRENT SPREADSHEET
    df_ratings['date: century'] = dates
    df_ratings.fillna(value=np.nan, inplace=True)
    df_ratings_sorted = df_ratings.sort_values(by=['date: century'])
    df_ratings_sorted.index = df_ratings_sorted['date: century']
    single_words[s['word']] = df_ratings_sorted
    plt.clf()
    plt.figure(figsize=(10, 16))
    g = seabornInstance.heatmap(df_ratings_sorted.iloc[:,:-1], 
                            cmap='Blues', 
                            vmin=0, 
                            #xticklabels=meanings_short,
                            annot=True)
    g.set_xticklabels(g.get_xticklabels(), wrap=True)
    g.set_xticklabels([textwrap.fill(e.replace('\xa0', ' '), 15, break_long_words=False) for e in df_ratings_sorted.columns], rotation =30)
    plt.title(s['word'])
    plt.savefig("heatmaps/"+s['word']+".png", bbox_inches='tight')
    plt.clf()
    plt.close()


# In[ ]:


len(spreadsheets) # should be 39


# In[ ]:


cumulative_df = pd.DataFrame(spreadsheets)


# ## Read separately 'virtus' annotations

# In[ ]:


virtus_paths = [f for f in os.listdir(os.path.join("..", "..", "virtus")) if f[-5:]=='.xlsx']
print(virtus_paths)


# In[ ]:


virtus_spreadsheets = []


# In[ ]:


for path in virtus_paths:
    printsep('-|-|')
    print('Working with the file ' + path)
    s = {} # will write all information from the spreadsheet to this dict
    split = re.split('_', path)
    s['annotator key'] = split[3][:-6]
    s['word'] = split[2]
    # OPEN FILE
    df = read_excel(os.path.join("..", "..", "virtus", path),
                                  sheet_name=0, encoding='utf-8', dtype=object)
    df.columns = [name.lower() for name in df.columns]
    df.replace('', np.NaN, inplace=True)
    df = df.dropna(how='all')
    print(df.tail())
    #s['file'] = df # not sure i need that
    # GET MEANINGS AND THEIR NUMBER
    column_headings = list(df.columns)
    print(column_headings)
    try:
        x = column_headings.index('right context')+1 #where metadata and test sentence ends
        y = column_headings.index('comments') #the last column (also checks whether it is "comments")
        meanings = [m.replace('\xa0', ' ') for m in column_headings[x:y]] #dealing with the \xa0 character
    except ValueError:
        print("No such columns in the spreadsheet.")
        meanings = []
        break
    print('Extracted meanings are:')
    print(meanings)
    s['meanings'] = meanings
    #add number of meanings to the dictionary
    s['number of meanings'] = len(meanings)
    #print(s['cells'])
    # FIND COMMENTS, COUNT THE NUMBER
    comments = df.dropna()
    print(comments)
    s['comments'] = comments
    s['number of comments'] = comments.shape[0]
    # SELECT A SUBSET WITH ONLY MEANINGS AND RATINGS + INDEX
    df_ratings = df.iloc[:, x:y]
    df_ratings = normalize_ratings(df_ratings)
    s['rows'] = df_ratings.shape[0]
    s['cells'] = s['rows'] * s['number of meanings']
    print(df_ratings.head())
    printsep('-')
    # COUNT UNIQUE PATTERNS OF ANNOTATION
    all_rating_patterns = []
    for i in range(df_ratings.shape[0]):
        pattern = str((sorted(df_ratings.iloc[i].to_list())))
        all_rating_patterns.append(pattern)
    unique_rating_patterns = set(all_rating_patterns)
    rating_patterns_counted = {}
    for p in all_rating_patterns:
        if p in rating_patterns_counted.keys():
            rating_patterns_counted[p] += 1
        else:
            rating_patterns_counted[p] = 1
    #pp.pprint(rating_patterns_counted)
    #print(rating_patterns_counted)
    # turn dictionary into a Series, then DataFrame (there must be a better way, but this works so far...)
    rating_patterns_counted = pd.Series(rating_patterns_counted)
    rating_patterns_counted = pd.DataFrame({'pattern': rating_patterns_counted.index, 'count': rating_patterns_counted.values})      # result
    print("Unique rating patterns:")
    pp.pprint(rating_patterns_counted)
    s['rating patterns'] = rating_patterns_counted
    printsep('-')
    # COUNT ALL VALUES ACCROSS THE DATAFRAME
    # stack all columns, do value_counts()
    arc = df_ratings.stack().value_counts()
    # add absolute numbers and their relative frequency to a dataframe
    arc = pd.DataFrame({'rating':arc.index,
                                        'absolute number':arc.values,
                                        'frequency': arc.values / s['cells']})   # result
    print("All ratings:")
    print(arc)
    try:
        fours = arc.loc[arc['rating'] == 4, 'absolute number'].item()
    except ValueError:
        fours = 0
    try:
        threes = arc.loc[arc['rating'] == 3, 'absolute number'].item()
    except ValueError:
        threes = 0
    try:
        twos = arc.loc[arc['rating'] == 2, 'absolute number'].item()
    except ValueError:
        twos = 0
    conf =  fours / (fours + threes + twos)
    s['individual ratings frequency'] = arc
    s['confidence'] = conf
    s['average'] = (fours * 4 + threes * 3 + twos * 2) / ((fours + threes + twos) * 4)
    s['ratings'] = df_ratings
    # SHOW RESULTING DICTIONARY
    pp.pprint(s)
    # ADD THIS WHOLE THING TO THE LIST OF DATA ON SPREADSHEETS
    virtus_spreadsheets.append(s)


# ### Create a 'virtus' heatmap

# In[ ]:


temp = df_ratings
for s in virtus_spreadsheets[:-1]:
    temp += s['ratings']
temp


# In[ ]:


temp.fillna(value=np.nan, inplace=True)
temp = temp.div(4)
plt.clf()
plt.figure(figsize=(10, 16))
g = seabornInstance.heatmap(temp, 
                            cmap='Blues', 
                            vmin=0, 
                            #xticklabels=meanings_short,
                            annot=True
                           )
g.set_xticklabels([textwrap.fill(e.replace('\xa0', ' '), 15, break_long_words=False) for e in df_ratings.columns], rotation =30)
plt.title(s['word']) # + ' : ' +s['annotator'])
plt.savefig(s['word'], bbox_inches='tight') # + ' : ' +s['annotator']+".png", )
plt.clf()
plt.close()


# In[ ]:


virtus_df = pd.DataFrame(virtus_spreadsheets)


# In[ ]:


# virtus_df.head()


# # Data analysis

# ## Annotation 'styles': grouping data by annotator

# Dataframe columns: 'annotator', 'word', 'rows', 'meanings', 'number of meanings', 'cells', 'comments', 'number of comments', 'rating patterns', 'individual ratings frequency'<br>

# In[ ]:


ann_list = []


# In[ ]:


cumulative_df['annotator key'].unique()


# In[ ]:


for annotator in cumulative_df['annotator key'].unique():
    ann = {}
    print('.'*15 + annotator + '.'*15)
    ann['annotator'] = annotator
    ann_df = cumulative_df[cumulative_df['annotator key'] == annotator]
    ann['words annotated'] = len(ann_df)
    ann['cells in total'] = sum(ann_df['cells'])
# calculate frequency of individual ratings
    irf = pd.concat(ann_df['individual ratings frequency'].to_list(), ignore_index=True).iloc[:, :2]
    printsep('//')
    print(irf)
    irf = irf.groupby(['rating']).sum()
    irf['frequency'] = irf['absolute number'] / sum(irf['absolute number'])
    confidence = irf['frequency'][4]/(irf['frequency'][2]+irf['frequency'][3]+irf['frequency'][4])
    average = (irf['absolute number'][3]*3 + irf['absolute number'][2]*2 + irf['absolute number'][4]*4) / ((irf['absolute number'][4]+irf['absolute number'][3]+irf['absolute number'][2])*4)
    irf.at[len(irf)+1,'confidence'] = confidence
    irf.at[len(irf)+1,'average'] = average
    ann['average'] = average
    ann["confidence"] = confidence
    print(irf)
    ann['individual rating frequency'] = irf
# calculate individual patterns
    prns = pd.concat(ann_df['rating patterns'].to_list(), ignore_index=True)
    prns = prns.groupby(['pattern'], as_index=False).sum()
    prns['frequency'] = prns['count'] / sum(prns['count'])
    ann['patterns'] = prns
    print(prns)
# calculate sets of patterns
    prns['set'] = [str(set(ast.literal_eval(x))) for x in prns['pattern']]
    sets = prns.groupby(['set'], as_index=False).sum()
    ann['sets'] = sets
# append single annotator dictionary to a list of annotators
    ann_list.append(ann)
    printsep('//')


# In[ ]:


# WRITE ANNOTATOR DATA TO .XLSX
for a in ann_list:
    with pd.ExcelWriter(a['annotator'] +'_' + str(a['words annotated']) + '_words' '_style_export.xlsx') as writer:
        a['individual rating frequency'].to_excel(writer, sheet_name='individual rating frequency')
        a['patterns'].to_excel(writer, sheet_name='pattern frequency')
        a['sets'].to_excel(writer, sheet_name='sets used')


# In[ ]:


ann_list


# In[ ]:


len(ann_list) # should be 7


# In[ ]:


# CREATE ANNOTATION DATAFRAME
annotation_styles = pd.DataFrame(ann_list)
# add virtus data if it exists
annotation_styles.insert(3, 'confidence: virtus', virtus_df['confidence'])
annotation_styles.insert(4, 'average: virtus', virtus_df['average'])


# In[ ]:


annotation_styles.iloc[:, :7]


# ### Comparison plots: test task vs. annotation task

# In[ ]:


plt.close()
plt.clf()
annotation_styles.plot.bar(figsize=(18,5), 
                           x='annotator', 
                           y=['confidence: virtus', 'confidence'],
                          cmap='Set1',
                          )
filename = 'virtus-versus-task confidence.png'
plt.savefig('confidence analysis/'+filename, bbox_inches='tight',transparent=True)


# In[ ]:


plt.close()
plt.clf()
annotation_styles.plot.bar(figsize=(18,5), 
                           x='annotator', 
                           y=['average: virtus', 'average'],
                          cmap='Set2')
filename = 'virtus-versus-task average.png'
plt.savefig('confidence analysis/'+filename, bbox_inches='tight', transparent=True)


# ### Additional calculation: number of all possible patterns

# In[ ]:


# counting the number of different patterns / number of possible combinations 
from scipy.special import comb
cumulative_df['count of patterns'] = cumulative_df['rating patterns'].apply(lambda x: x.shape[0])
cumulative_df['count of all possible patterns'] = comb(5, cumulative_df['number of meanings'], repetition=True)
cumulative_df['% of possible pp used'] = cumulative_df['count of patterns'] / cumulative_df['count of all possible patterns']


# In[ ]:


for i in range(cumulative_df.shape[0]):
    print(cumulative_df['word'][i])
    print('this word has ' + str(cumulative_df['number of meanings'][i]) + ' meanings')
    pp.pprint(cumulative_df['meanings'][i])
    printsep('.')


# ### Dataframe of confidences

# In[ ]:


confidence_df = cumulative_df.loc[:, ['annotator key', 'word', 'number of meanings', 'confidence', 'average', 'count of patterns','count of all possible patterns', '% of possible pp used']]


# In[ ]:


annotators_meanings_conf = confidence_df[['annotator key', 'word', 'number of meanings', 'confidence', 'average']]
annotators_meanings_conf


# In[ ]:


annotation_styles


# ### Analysis and weighting of the impact of the number of meanings on confidence

# In[ ]:


colors = {'A1': '#377eb8', 
          'A2': '#ff7f00',
          'A3': '#4daf4a',
          'A4':  '#f781bf',
          'A5': '#a65628',
          'A6': '#984ea3',
          'A7': '#999999',
         }

'''
'#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'
                  SOURCE: https://gist.github.com/thriveth/8560036
'''

markers = {'A1': 'o', 
          'A2': 'x',
          'A3': '+',
          'A4':  'v',
          'A5': '*',
          'A6': '^',
          'A7': 's',
         } 


# In[ ]:


plt.clf()
plt.rcParams['figure.figsize'] = [10, 10]
fig = plt.figure()
ax = fig.gca()
for k in cumulative_df['annotator key'].unique():
    plot_scatter(annotators_meanings_conf.loc[annotators_meanings_conf['annotator key']==k], 'number of meanings', 'confidence', 'annotator key', 
             filename='confidence analysis/meanings_confidence_annotators.png', 
             a=1,
            c=colors[k],
             m=markers[k], 
            )


# In[ ]:


grouped = cumulative_df.groupby(['number of meanings'])


# In[ ]:


n_of_meanings = cumulative_df['number of meanings'].unique()


# In[ ]:


meanings_and_patterns = {}


# In[ ]:


for i in range(2,8): 
    m = grouped.get_group(i)
    meanings_and_patterns[i] = pd.concat([m['annotator key'],m['word'],m['rating patterns'],m['individual ratings frequency']], axis=1)


# In[ ]:


confidence_meanings = {}
average_meanings = {}


# In[ ]:


for key in meanings_and_patterns:
    # the value is a df
    nr_df = meanings_and_patterns[key]
    w_and_a = nr_df.iloc[:, 0:2]
    prns = pd.concat(nr_df['rating patterns'].to_list(), ignore_index=True)
    prns = prns.groupby(['pattern'], as_index=False).sum()
    prns['frequency'] = prns['count'] / sum(prns['count'])
    irfs = pd.concat(nr_df['individual ratings frequency'].to_list(), ignore_index=True)
    irfs = irfs.groupby(['rating'], as_index=False).sum()
    irfs['frequency'] = irfs['absolute number']/sum(irfs['absolute number'])
    confidence = irfs['frequency'][4]/(irfs['frequency'][2]+irfs['frequency'][3]+irfs['frequency'][4])
    average = (irfs['absolute number'][4]*4 + irfs['absolute number'][3]*3 + irfs['absolute number'][2]*2) / ((irfs['absolute number'][2]+irfs['absolute number'][3]+irfs['absolute number'][4])*4)
    irfs.at[len(irfs),'confidence'] = confidence
    irfs.at[len(irfs),'average'] = average
    confidence_meanings[key] = confidence
    average_meanings[key] = average
    with pd.ExcelWriter('{} meanings.xlsx'.format(key)) as writer:
        w_and_a.to_excel(writer, sheet_name='annotators and words')
        irfs.to_excel(writer, sheet_name='rating frequency')
        prns.to_excel(writer, sheet_name='pattern frequency')


# In[ ]:


average_meanings


# In[ ]:


# the outlier 'jus' is taken out; confidence '6a' stores the value with the outlier
confidence_meanings['6a']=confidence_meanings[6]


# In[ ]:


confidence_meanings[6]=0.563003 # average without 'jus'
confidence_meanings


# In[ ]:


by_meanings = {}

for key in average_meanings.keys():
    by_meanings[key]=[average_meanings[key], confidence_meanings[key]]

df_by_meanings = pd.DataFrame(by_meanings, index=['meanings average', 'meanings confidence']).transpose()


# In[ ]:


df_by_meanings.loc[2, 'meanings confidence'] 


# In[ ]:


plt.close()
plt.clf()
df_by_meanings['meanings confidence'].plot(figsize=(18,5),
                                      x='number of meanings', 
                                      y='meanings confidence')
filename = 'confidence by number of meanings_new6.png'
plt.savefig(filename, bbox_inches='tight', transparent=True)


# In[ ]:


annotators_confidence = {dict["annotator"]:dict["confidence"] for dict in ann_list}


# In[ ]:


annotators_meanings_conf = pd.merge(annotators_meanings_conf, df_by_meanings, left_on='number of meanings', right_index=True)


# In[ ]:


annotators_meanings_conf.sort_index(inplace=True)
annotators_meanings_conf.head()


# In[ ]:


# ADDING WEIGHTING COEFFICIENT
k3 = 1
annotators_meanings_conf['coefficient']=1/(annotators_meanings_conf['meanings confidence']/annotators_meanings_conf['meanings confidence'][0])
annotators_meanings_conf['meanings confidence'][0]
annotators_meanings_conf['weighted'] = annotators_meanings_conf['confidence'] * annotators_meanings_conf['coefficient']


# In[ ]:


virtus_coeff = float(annotators_meanings_conf.loc[annotators_meanings_conf['number of meanings']==6, ['coefficient']].iloc[0])


# In[ ]:


virtus_coeff


# In[ ]:


weighted_df = pd.DataFrame(index=annotators_confidence.keys())
weighted_df['weighted confidence'] = np.nan


# In[ ]:


weighted_df


# In[ ]:


for annotator in annotators_confidence.keys():
    #print(annotator)
    annotator_df = annotators_meanings_conf[annotators_meanings_conf['annotator key'] == annotator]
    #print(annotator_df)
    weighted_confidence = annotator_df['weighted'].sum()/len(annotator_df)
    #print(weighted_confidence)
    weighted_df.loc[[annotator], ['weighted confidence']] = weighted_confidence


# In[ ]:


weighted_df.reset_index(level=0, inplace=True)
weighted_df = weighted_df.rename(columns={'index': 'annotator'})


# In[ ]:


weighted_df


# In[ ]:


annotation_styles = annotation_styles.merge(weighted_df, on='annotator')
annotation_styles['weighted confidence: virtus'] = virtus_coeff * annotation_styles['confidence: virtus']
annotation_styles[['annotator', 'words annotated', 'cells in total', 'confidence: virtus', 'weighted confidence: virtus', 'confidence', 'weighted confidence']]


# #### Plots: annotators' confidence scores before and after weighting

# In[ ]:


plt.close()
plt.clf()
annotation_styles.plot.bar(figsize=(18,5), 
                           x='annotator', 
                           y=[
                               'confidence: virtus', 
                               'weighted confidence: virtus',
                               'confidence', 
                               'weighted confidence',
                           ])
filename = 'confidence analysis/virtus-average-vs-weighted-confidence_new6.png'
plt.savefig(filename, bbox_inches='tight', transparent=True)


# In[ ]:


plt.close()
plt.clf()
annotation_styles.plot.bar(figsize=(18,5), 
                           x='annotator', 
                           y=[
                               #'confidence: virtus', 
                               'confidence', 
                               #'weighted confidence: virtus',
                               'weighted confidence',
                           ])
filename = 'confidence analysis/average-vs-weighted-confidence_new6.png'
plt.savefig(filename, bbox_inches='tight',transparent=True)


# In[ ]:


annotators_meanings_conf = annotators_meanings_conf.rename(columns={'weighted':'weighted confidence'})


# #### Plots: scatter plots of all annotated words with weighted confidence values

# In[ ]:


plt.clf()
plt.rcParams['figure.figsize'] = [10, 10]
fig = plt.figure()
ax = fig.gca()
for k in cumulative_df['annotator key'].unique():
    plot_scatter(annotators_meanings_conf.loc[annotators_meanings_conf['annotator key']==k], 'number of meanings', 'weighted confidence', 'word', 
             filename='confidence analysis/meanings_weighted confidence_words.png', 
             a=1,
            c=colors[k],
             #m=markers[k],                 
            )


# In[ ]:


plt.clf()
plt.rcParams['figure.figsize'] = [10, 10]
fig = plt.figure()
ax = fig.gca()
for k in cumulative_df['annotator key'].unique():
    plot_scatter(annotators_meanings_conf.loc[annotators_meanings_conf['annotator key']==k], 'number of meanings', 'weighted confidence', 'annotator key', 
             filename='scatter_plots/meanings_confidence_annotators_weighted.png', 
             a=1,
            c=colors[k],
             m=markers[k],
                 
            )


# In[ ]:


annotators_meanings_conf[['confidence', 
    'weighted confidence']].plot(kind='box', #title='comparison of non-weighted and weighted confidences'
                                )
plt.savefig('confidence analysis/weighed-confidences-boxplot-new6.png')
    


# ## Adding new data to the annotation_styles dataframe

# Now we have two values: 
# - expected confidence of a specific annotator
# - expected confidence for a word with N meanings

# In[ ]:


float(annotators_meanings_conf[annotators_meanings_conf['word']=='acerbus']['weighted confidence'])


# In[ ]:


annotation_styles.columns


# In[ ]:


confidence_df['expected confidence: annotator'] = confidence_df['annotator key'].apply(lambda x: annotators_confidence[x])


# In[ ]:


confidence_df['expected weighted confidence: annotator'] = confidence_df['annotator key'].apply(lambda x: float(annotation_styles[annotation_styles['annotator']==x]['weighted confidence']) )


# In[ ]:


confidence_df['virtus confidence: annotator'] = confidence_df['annotator key'].apply(
    lambda x: annotation_styles.loc[annotation_styles['annotator']==x, 'confidence: virtus'].values[0])
confidence_df['virtus weighted confidence: annotator'] = confidence_df['annotator key'].apply(
    lambda x: annotation_styles.loc[annotation_styles['annotator']==x, 'weighted confidence: virtus'].values[0])


# In[ ]:


confidence_df['expected confidence: nr of meanings'] = confidence_df['number of meanings'].apply(lambda x: confidence_meanings[x])
confidence_df['expected average: nr of meanings'] = confidence_df['number of meanings'].apply(lambda x: average_meanings[x])
confidence_df['weighted confidence'] = confidence_df['word'].apply(lambda x: float(annotators_meanings_conf[annotators_meanings_conf['word']==x]['weighted confidence']))


# In[ ]:


confidence_df['diff to annotator'] = confidence_df['expected confidence: annotator']-confidence_df['confidence']
confidence_df['diff to meanings'] = confidence_df['expected confidence: nr of meanings']-confidence_df['confidence']


# In[ ]:


confidence_df


# ## Comparison of confidence and the semantic qualities of annotated words

# In[ ]:


words_data = read_excel(os.path.join("..", "..", 'Words_qualities.xlsx'), sheet_name=0, encoding='utf-8', dtype=object)


# In[ ]:


words_data


# ### Comparison with the structure of the dictionary entry (Lewis and Short)
# Data on the meanings was from L&S was collected manually and stored as a string (e.g. 'I, I.B.1, I.B.3, I.B.4, II'), in which each item corresponds to the position of the selected meaning within the hierarchy of the L&S entry for the words. 

# In[ ]:


lewis_and_short = words_data[['word','L&S hierarchy']]


# In[ ]:


class Node:

    def __init__(self, name):
        self.name = name
        self.children = []
        
    def list_children(self):
        if len(self.children) == 0:
            return ''
        result_string = ': [ '
        for child in self.children:
            result_string += child.to_string() + ' '
        result_string += ']'
        return result_string
        
    
    def to_string(self):
        return self.name + self.list_children()
    


# In[ ]:


def node_parser(tokens, parent_node): # tokens is a list
    if len(tokens) == 0:
        return
    
    child_node = None
    for child in parent_node.children:
        if child.name == tokens[0]:
            child_node = child
    if child_node == None:
        child_node = Node(tokens[0])
        parent_node.children.append(child_node)
        
    node_parser(tokens[1:], child_node)
        
        
def hierarchy(string):
    string = ''.join(string.split())
    print(string)
    x = string.split(',') # x is a list
    print(len(x)) # number of meanings
    root = Node('ø')
    for i in range(len(x)):
        tokens = x[i].split('.') 
        node_parser(tokens, root)
    
    print(root.to_string())
    return root


# In[ ]:


test_string = 'I, I.B.1, I.B.3, I.B.4, II'
hierarchy(test_string)


# In[ ]:


def find_splits(node):
    print(node.to_string())
    explicit_splits = 0
    if len(node.children) == 1:
        pass
    else: 
        explicit_splits = len(node.children)
    
    implicit_splits = 0
    for child in node.children:
        implicit_splits += find_splits(child)
    
    return explicit_splits + implicit_splits
    

def tree_complexity(string):
    root = hierarchy(string)
    print('in tree_complexity())' + root.to_string())
    return find_splits(root)


# In[ ]:


print(tree_complexity(test_string))


# In[ ]:


test = tree_complexity('I, II.A, II.B, II.C.1, II.C.2.a, II.C.2.b, II.B.2')


# In[ ]:


words_data['L&S complexity'] = words_data['L&S hierarchy'].map(tree_complexity)


# In[ ]:


words_data


# In[ ]:


confidence_df = pd.merge(left=confidence_df, right=words_data, how='right', left_on=['word', 'number of meanings'], right_on=['word', 'number of meanings'])


# In[ ]:


print(confidence_df[confidence_df['word']=='jus']['expected confidence: nr of meanings'])


# In[ ]:


confidence_df.columns


# In[ ]:


confidence_df.to_excel('Confidence comparison_new6.xlsx')


# In[ ]:


confidence_df.head()


# ### Visualisation of the relationship between confidence and suggested influencing quality for each words

# In[ ]:


plot_scatter(confidence_df, 'weighted confidence', 'confidence', 'word')


# In[ ]:


plot_scatter(confidence_df, 'expected weighted confidence: annotator', 'confidence', 'word')


# In[ ]:


plot_scatter(confidence_df, 'sources (Fr)', 'confidence', 'word')


# In[ ]:


plot_scatter(confidence_df, '% of possible pp used', 'confidence', 'word')


# In[ ]:


plot_scatter(confidence_df, 'count of patterns', 'confidence', 'word')


# In[ ]:


plot_scatter(confidence_df, 'count of patterns', 'average', 'word')


# In[ ]:


plot_scatter(confidence_df, 'L&S complexity', 'confidence', 'word')


# In[ ]:


plot_scatter(confidence_df, 'L&S complexity', 'number of meanings', 'word')


# In[ ]:


plot_scatter(confidence_df, 'POS', 'confidence', None)


# In[ ]:


#ax = confidence_df.plot.scatter(x='confidence', y='word', color='blue', label='real confidence')
#confidence_df.plot.scatter(x='expected confidence: annotator', y='word', color='green', ax=ax, label='annotator confidence')
#confidence_df.plot.scatter(x='expected confidence: nr of meanings', y='word', color='cyan', ax=ax, label='meanings confidence', )


# In[ ]:


'''
plt.rcParams['figure.figsize'] = [20, 4]
pl = confidence_df.plot.scatter(x='word', y='diff to annotator', color='blue', label='diff to annotator')
confidence_df.plot.scatter(x='word', y='diff to meanings', color='red', label='diff to meanings', ax=pl)
plt.xticks(rotation=45)
plt.axhline(y=0.1, c='gray')
plt.axhline(y=-0.1, c='gray')
plt.axhline()
'''


# In[ ]:


'''
plt.rcParams['figure.figsize'] = [20, 4]
pl = confidence_df.loc[confidence_df['target or control?_x']=='target'].plot.scatter(x='word', y='confidence', color='green', label='confidence: target words')
confidence_df.loc[confidence_df['target or control?_x']=='control'].plot.scatter(x='word', y='confidence', color='black', label='confidence: control words', ax=pl)
plt.xticks(rotation=45)
plt.axhline(y=0.5, c='gray')
#plt.axhline(y=0.75, c='gray')
plt.axhline()
plt.savefig('words_confidences_target_or_control.png', bbox_inches='tight')
'''


# In[ ]:


'''
plt.rcParams['figure.figsize'] = [20, 4]
pl = confidence_df.loc[confidence_df['target or control?_x']=='target'].plot.scatter(x='word', y='weighted confidence', color='green', label='confidence: target words')
confidence_df.loc[confidence_df['target or control?_x']=='control'].plot.scatter(x='word', y='weighted confidence', color='black', label='confidence: control words', ax=pl)
plt.xticks(rotation=45)
plt.axhline(y=0.5, c='gray')
#plt.axhline(y=0.75, c='gray')
plt.axhline()
plt.savefig('words_confidences_target_or_control_weighted.png', bbox_inches='tight')
'''


# In[ ]:


plt.rcParams['figure.figsize'] = [20, 4]
pl = confidence_df.plot.scatter(x='word', 
                                y='weighted confidence', 
                                color='purple', 
                                #label='confidence: all words'
                               )
plt.xticks(rotation=45)
#plt.axhline(y=0.5, c='gray')
#plt.axhline(y=0.75, c='gray')
plt.axhline()
plt.savefig('confidence analysis/words_confidences_all_weighted.png', bbox_inches='tight')


# In[ ]:


plt.rcParams['figure.figsize'] = [20, 4]
pl = confidence_df.plot.scatter(x='word', 
                                y='confidence', 
                                color='purple', 
                                #label='confidence: all words'
                               )
plt.xticks(rotation=45)
#plt.axhline(y=0.5, c='gray')
#plt.axhline(y=0.75, c='gray')
plt.axhline()
plt.savefig('confidence analysis/words_confidences_all_notweighted.png', bbox_inches='tight')


# In[ ]:


plt.rcParams['figure.figsize'] = [20, 4]
pl = confidence_df.plot.scatter(x='word', 
                                y='average', 
                                color='teal', 
                                #label='confidence: all words'
                               )
plt.xticks(rotation=45)
#plt.axhline(y=0.5, c='gray')
#plt.axhline(y=0.75, c='gray')
#plt.axhline()
plt.savefig('confidence analysis/words_averages_all.png', bbox_inches='tight')


# ## Plotting single words

# In[ ]:


# single_words is a dictionary of dataframes containing cleaned up annotation data for every word in the task
single_words


# In[ ]:


def context_confidence(row):
    context = row.to_list()[:-1]
    fours = 0
    others = 0
    for rating in context:
        if rating == 4.0:
            fours += 1
        if rating == 3.0 or rating == 2.0:
            others += 1
    try: 
        return(fours/(fours+others))
    except ZeroDivisionError:
        return(0)


# In[ ]:


def context_average(row):
    context = row.to_list()[:-1]
    fours = 0
    others = 0
    for rating in context:
        if rating == 4.0:
            fours += 1
        if rating == 3.0 or rating == 2.0:
            others += 1
    try: 
        return(fours/(fours+others))
    except ZeroDivisionError:
        return(0)


# In[ ]:


confidence_df.head()


# In[ ]:


def single_word_analysis(analysed_word):    
    word_df = single_words[analysed_word]
    word_conf = confidence_df.loc[confidence_df['word']==analysed_word]['confidence'].to_list()[0]
    word_conf_weighted = confidence_df.loc[confidence_df['word']==analysed_word]['weighted confidence'].to_list()[0]
    word_average = confidence_df.loc[confidence_df['word']==analysed_word]['average'].to_list()[0]
    word_df.index = range(len(word_df))
    word_df['confidence'] = word_df.apply(context_confidence, axis=1)
    plt.rcParams['figure.figsize'] = [20, 4]
    pl = word_df.plot(y='confidence', linestyle="",marker="o") #(x='date: century', y='confidence', color='purple', label='confidence per context')
    #plt.xticks(rotation=45)
    plt.axhline(y= word_conf, c='gray')
    plt.savefig('words visualised/all contexts visualisation/' + analysed_word +'_all_contexts_visualised.png', bbox_inches='tight')
    plt.clf()


# In[ ]:


def single_word_diachronic_plot(analysed_word):
    print(analysed_word)
    word_df=single_words[analysed_word]
    plt.rcParams['figure.figsize'] = [20, 4]
    pivot = pd.pivot_table(word_df.iloc[:, :-1].replace([1],0),
                       index='date: century',
                       aggfunc=np.average
    )
    pivot = pivot.reindex(np.arange(-2, 20.5,0.5))
    pivot = pivot.div(pivot.sum(1), axis=0)
    pivot.plot(kind='bar', stacked=True)
    filename='{0}-diachronic-plot'.format(analysed_word)
    plt.savefig('words visualised/diachronic plots/'+filename, bbox_inches='tight')
    plt.clf()


# In[ ]:


def single_word_diachronic_plot_unstacked(analysed_word):
    print(analysed_word)
    word_df=single_words[analysed_word]
    pivot = pd.pivot_table(word_df.iloc[:, :-1].replace([1],0),
                       index='date: century',
                       aggfunc=np.average
    )
    plt.rcParams['figure.figsize'] = [20, 4]
    pivot = pivot.reindex(np.arange(-2, 20.5,0.5))
    #pivot = pivot.div(pivot.sum(1), axis=0)
    pivot.plot(kind='bar', stacked=True)
    filename='{0}-diachronic-plot-unstacked'.format(analysed_word)
    plt.savefig('words visualised/diachronic plots/'+filename, bbox_inches='tight')
    plt.clf()


# In[ ]:


for word in single_words.keys():
    single_word_diachronic_plot(word)


# In[ ]:


for word in single_words.keys():
    single_word_diachronic_plot_unstacked(word)


# In[ ]:


for word in single_words.keys():
    single_word_analysis(word)


# # Linear regression models

# ## Looking for explanations of confidence

# In[ ]:


pl = confidence_df.plot.scatter(x='confidence', y='expected confidence: nr of meanings', figsize=[7,7])
pl.plot([0, 1], [0, 1])


# In[ ]:


pl = confidence_df.plot.scatter(x='confidence', y='expected confidence: annotator', figsize=[7,7])
pl.plot([0, 1], [0, 1])


# In[ ]:


pl = confidence_df.plot.scatter(x='expected confidence: nr of meanings', y='expected confidence: annotator', figsize=[7,7])
pl.plot([0,2, 1,2], [0,2, 1,2])


# In[ ]:


confidence_df


# In[ ]:


models_results = []


# ## Model #1: based on expected confidences

# In[ ]:


model = '#1'
predictors = ['expected confidence: annotator', 'expected confidence: nr of meanings']


# In[ ]:


X = confidence_df[predictors].values


# In[ ]:


y = confidence_df['confidence'].values


# In[ ]:


seabornInstance.distplot(confidence_df['confidence'])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=range(len(y_test)))


# In[ ]:


df.plot(kind='bar')


# ### Test 1: Mean absolute error should be within the 10% range of the mean confidence value.

# In[ ]:


confidence_df['confidence'].mean()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ### Test 2: R squared is in the interval (0, 1), the greater the better

# In[ ]:


metrics.r2_score(y_test, y_pred)


# In[ ]:


metrics.explained_variance_score(y_test, y_pred) # maximum value is 1 and is desirable


# In[ ]:


metrics.max_error(y_test, y_pred) 


# In[ ]:


metrics.median_absolute_error(y_test, y_pred)


# In[ ]:


# add result to a dictionary
models_results.append({
    'model': model, 
    'predictors': predictors, 
    'MAE': metrics.mean_absolute_error(y_test, y_pred),
    'MSE': metrics.mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    'r squared': metrics.r2_score(y_test, y_pred),
    'explained variance score': metrics.explained_variance_score(y_test, y_pred),
    'max error': metrics.max_error(y_test, y_pred)
})


# ## Model #1.a: based on expected annotator & meanings confidences +  virtus confidences (only for the 4 annotators who annotated virtus)

# In[ ]:


model = '#1.a'
predictors = ['virtus confidence: annotator', 'expected confidence: annotator', 'expected confidence: nr of meanings']


# In[ ]:


X = confidence_df[predictors].dropna().values


# In[ ]:


y = confidence_df[['virtus confidence: annotator', 'expected confidence: annotator', 'expected confidence: nr of meanings', 'confidence']].dropna().values


# In[ ]:


len(y)


# In[ ]:


seabornInstance.distplot(confidence_df[['virtus confidence: annotator', 'expected confidence: annotator', 'expected confidence: nr of meanings', 'confidence']].dropna()['confidence'])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


y_pred[:, 0]


# In[ ]:


y_test


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=range(len(y_test)))


# In[ ]:


df.plot(kind='bar')


# ### Test 1: Mean absolute error should be within the 10% range of the mean confidence value.

# In[ ]:


confidence_df.dropna()['confidence'].mean()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ### Test 2: R squared 

# In[ ]:


metrics.r2_score(y_test, y_pred)


# In[ ]:


metrics.explained_variance_score(y_test, y_pred) # maximum value is 1 and is desirable


# In[ ]:


metrics.max_error(y_test, y_pred) 


# In[ ]:


metrics.median_absolute_error(y_test, y_pred)


# In[ ]:


# add result to a dictionary
models_results.append({
    'model': model, 
    'predictors': predictors, 
    'MAE': metrics.mean_absolute_error(y_test, y_pred),
    'MSE': metrics.mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    'r squared': metrics.r2_score(y_test, y_pred),
    'explained variance score': metrics.explained_variance_score(y_test, y_pred),
    'max error': metrics.max_error(y_test, y_pred)
})


# ## Model #1.b: based only on number of meanings +  virtus confidences 
# (only for the 4 annotators who did virtus)

# In[ ]:


model = '#1.b'
predictors = ['number of meanings', 'virtus confidence: annotator']


# In[ ]:


X = confidence_df[predictors].dropna().values


# In[ ]:


y = confidence_df.dropna()['confidence'].values


# In[ ]:


len(X) == len(y)


# In[ ]:


seabornInstance.distplot(confidence_df.dropna()['confidence'])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=range(len(y_test)))


# In[ ]:


df.plot(kind='bar')


# ### Test 1: Mean absolute error should be within the 10% range of the mean confidence value.

# In[ ]:


confidence_df.dropna()['confidence'].mean()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ### Test 2: R squared is in the interval (0, 1), the greater the better 

# In[ ]:


metrics.r2_score(y_test, y_pred)


# In[ ]:


metrics.explained_variance_score(y_test, y_pred) # maximum value is 1 and is desirable


# In[ ]:


metrics.max_error(y_test, y_pred) 


# In[ ]:


metrics.median_absolute_error(y_test, y_pred)


# In[ ]:


# add result to a dictionary
models_results.append({
    'model': model, 
    'predictors': predictors, 
    'MAE': metrics.mean_absolute_error(y_test, y_pred),
    'MSE': metrics.mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    'r squared': metrics.r2_score(y_test, y_pred),
    'explained variance score': metrics.explained_variance_score(y_test, y_pred),
    'max error': metrics.max_error(y_test, y_pred)
})


# ## Model #1.c: based on number of meanings +  virtus confidences + count of patterns
# (only for the 4 annotators who did virtus)

# In[ ]:


model = '#1.c'
predictors = ['number of meanings', 'virtus confidence: annotator', 'count of patterns']


# In[ ]:


X = confidence_df[predictors].dropna().values


# In[ ]:


y = confidence_df.dropna()['confidence'].values


# In[ ]:


len(X) == len(y)


# In[ ]:


seabornInstance.distplot(confidence_df.dropna()['confidence'])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=range(len(y_test)))


# In[ ]:


df.plot(kind='bar')


# ### Test 1: Mean absolute error should be within the 10% range of the mean confidence value.

# In[ ]:


confidence_df.dropna()['confidence'].mean()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ### Test 2: R squared is in the interval (0, 1), the greater the better

# In[ ]:


metrics.r2_score(y_test, y_pred)


# In[ ]:


metrics.explained_variance_score(y_test, y_pred) # maximum value is 1 and is desirable


# In[ ]:


metrics.max_error(y_test, y_pred) 


# In[ ]:


metrics.median_absolute_error(y_test, y_pred)


# In[ ]:


# add result to a dictionary
models_results.append({
    'model': model, 
    'predictors': predictors, 
    'MAE': metrics.mean_absolute_error(y_test, y_pred),
    'MSE': metrics.mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    'r squared': metrics.r2_score(y_test, y_pred),
    'explained variance score': metrics.explained_variance_score(y_test, y_pred),
    'max error': metrics.max_error(y_test, y_pred)
})


# ## Model #2: based on (1) annotator confidence; (2) number of meanings 

# In[ ]:


model = '#2'
predictors = ['expected confidence: annotator', 'number of meanings']


# In[ ]:


X = confidence_df[predictors].values
#X = confidence_df['expected confidence: annotator'].values.reshape(-1,1)


# In[ ]:


y = confidence_df['confidence'].values


# In[ ]:


seabornInstance.distplot(confidence_df['confidence'])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=range(len(y_test)))


# In[ ]:


df.plot(kind='bar')


# ### Test 1: Mean absolute error should be within the 10% range of the mean confidence value.

# In[ ]:


confidence_df['confidence'].mean()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ### Test 2: R squared is in the interval (0, 1), the greater the better

# In[ ]:


metrics.r2_score(y_test, y_pred)


# In[ ]:


# add result to a dictionary
models_results.append({
    'model': model, 
    'predictors': predictors, 
    'MAE': metrics.mean_absolute_error(y_test, y_pred),
    'MSE': metrics.mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    'r squared': metrics.r2_score(y_test, y_pred),
    'explained variance score': metrics.explained_variance_score(y_test, y_pred),
    'max error': metrics.max_error(y_test, y_pred)
})


# # Model #3: Based on annotator and number of meanings

# In[ ]:


model = '#3'
predictors = ['annotator', 'number of meanings']


# In[ ]:


annotators = pd.get_dummies(cumulative_df['annotator key'])


# In[ ]:


annotators


# In[ ]:


train = pd.concat([cumulative_df, annotators], axis=1)


# In[ ]:


train


# In[ ]:


X = train[['A1', 'A2', 'A7', 'A6', 'A3', 'A5', 'A4']].values


# In[ ]:


y = train['confidence'].values


# In[ ]:


y


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=range(len(y_test)))


# In[ ]:


df.plot(kind='bar')


# ### Test 1: Mean absolute error should be within the 10% range of the mean confidence value.

# In[ ]:


train['confidence'].mean()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ### Test 2: R squared is in the interval (0, 1), the greater the better

# In[ ]:


metrics.r2_score(y_test, y_pred)


# In[ ]:


# add result to a dictionary
models_results.append({
    'model': model, 
    'predictors': predictors, 
    'MAE': metrics.mean_absolute_error(y_test, y_pred),
    'MSE': metrics.mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    'r squared': metrics.r2_score(y_test, y_pred),
    'explained variance score': metrics.explained_variance_score(y_test, y_pred),
    'max error': metrics.max_error(y_test, y_pred)
})


# > Interim results: models do not look too good. The best results were achieved by the model based on "annotator confidence" and "meanings confidence". 

# # Model #4: based on (1) count of patterns and (2) number of meanings
# **< Spoiler: the best R-squared so far >**

# In[ ]:


model = '#4'
predictors = ['count of patterns', 'number of meanings']


# In[ ]:


X = confidence_df[predictors].values


# In[ ]:


y = confidence_df['confidence'].values


# In[ ]:


seabornInstance.distplot(confidence_df['confidence'])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=range(len(y_test)))


# In[ ]:


df.plot(kind='bar')


# ### Test 1: Mean absolute error should be within the 10% range of the mean confidence value.

# In[ ]:


confidence_df['confidence'].mean()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ### Test 2: R squared is in the interval (0, 1), the greater the better

# In[ ]:


metrics.r2_score(y_test, y_pred)


# In[ ]:


metrics.explained_variance_score(y_test, y_pred) # maximum value is 1 and is desirable


# In[ ]:


metrics.max_error(y_test, y_pred) 


# In[ ]:


metrics.median_absolute_error(y_test, y_pred)


# In[ ]:


# add result to a dictionary
models_results.append({
    'model': model, 
    'predictors': predictors, 
    'MAE': metrics.mean_absolute_error(y_test, y_pred),
    'MSE': metrics.mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    'r squared': metrics.r2_score(y_test, y_pred),
    'explained variance score': metrics.explained_variance_score(y_test, y_pred),
    'max error': metrics.max_error(y_test, y_pred)
})


# # Model #5: based on (1) meaning clusters and (2) count of patterns
# 
# _*NB* the way of counting meaning clusters should be improved_

# In[ ]:


model = '#5'
predictors = ['count of patterns', 'meaning clusters']


# In[ ]:


X = confidence_df[['count of patterns', 'meaning clusters (D)']].values


# In[ ]:


y = confidence_df['confidence'].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=range(len(y_test)))


# In[ ]:


df.plot(kind='bar')


# ### Test 1: Mean absolute error should be within the 10% range of the mean confidence value.

# In[ ]:


confidence_df['confidence'].mean()


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ### Test 2: R squared is in the interval (0, 1), the greater the better

# In[ ]:


metrics.r2_score(y_test, y_pred)


# In[ ]:


metrics.explained_variance_score(y_test, y_pred) # maximum value is 1 and is desirable


# In[ ]:


metrics.max_error(y_test, y_pred) 


# In[ ]:


metrics.median_absolute_error(y_test, y_pred)


# In[ ]:


# add result to a dictionary
models_results.append({
    'model': model, 
    'predictors': predictors, 
    'MAE': metrics.mean_absolute_error(y_test, y_pred),
    'MSE': metrics.mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    'r squared': metrics.r2_score(y_test, y_pred),
    'explained variance score': metrics.explained_variance_score(y_test, y_pred),
    'max error': metrics.max_error(y_test, y_pred)
})


# > Model #5 performs worse than model #4
# 

# # Model #6: based on (1) number of sources and (2) count of patterns

# In[ ]:


model = '#6'
predictors = ['count of patterns', 'sources (Fr)']


# In[ ]:


confidence_df.columns


# In[ ]:


X = confidence_df[['count of patterns', 'sources (Fr)']].values


# In[ ]:


y = confidence_df['confidence'].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=range(len(y_test)))


# In[ ]:


df.plot(kind='bar')


# ### Test 1: Mean absolute error should be within the 10% range of the mean confidence value.

# In[ ]:


confidence_df['confidence'].mean() # mean confidence value = 0.6191 >> MAE should be not greater than 0.06


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ### Test 2: R squared is in the interval (0, 1), the greater the better

# In[ ]:


metrics.r2_score(y_test, y_pred)


# In[ ]:


metrics.explained_variance_score(y_test, y_pred) # maximum value is 1 and is desirable


# In[ ]:


metrics.max_error(y_test, y_pred) 


# In[ ]:


metrics.median_absolute_error(y_test, y_pred)


# In[ ]:


# add result to a dictionary
models_results.append({
    'model': model, 
    'predictors': predictors,
    'additional': 'random_state=42',
    'MAE': metrics.mean_absolute_error(y_test, y_pred),
    'MSE': metrics.mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    'r squared': metrics.r2_score(y_test, y_pred),
    'explained variance score': metrics.explained_variance_score(y_test, y_pred),
    'max error': metrics.max_error(y_test, y_pred)
})


# # Model #6a: based on number of meanings and complexity

# In[ ]:


model = '#6a'
predictors = ['sources (Fr)', 'L&S complexity']


# In[ ]:


confidence_df.columns


# In[ ]:


X = confidence_df[['number of meanings', 'L&S complexity']].values


# In[ ]:


y = confidence_df['confidence'].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=range(len(y_test)))


# In[ ]:


df.plot(kind='bar')


# ### Test 1: Mean absolute error should be within the 10% range of the mean confidence value.

# In[ ]:


confidence_df['confidence'].mean() # mean confidence value = 0.6094 >> MAE should be not greater than 0.06


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ### Test 2: R squared is in the interval (0, 1), the greater the better

# In[ ]:


metrics.r2_score(y_test, y_pred)


# In[ ]:


metrics.explained_variance_score(y_test, y_pred) # maximum value is 1 and is desirable


# In[ ]:


metrics.max_error(y_test, y_pred) 


# In[ ]:


metrics.median_absolute_error(y_test, y_pred)


# In[ ]:


# add result to a dictionary
models_results.append({
    'model': model, 
    'predictors': predictors,
    'additional': 'random_state=42',
    'MAE': metrics.mean_absolute_error(y_test, y_pred),
    'MSE': metrics.mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    'r squared': metrics.r2_score(y_test, y_pred),
    'explained variance score': metrics.explained_variance_score(y_test, y_pred),
    'max error': metrics.max_error(y_test, y_pred)
})


# # Model #7: based on (1) number of sources and (2) number of meanings

# In[ ]:


model = '#7'
predictors = ['number of meanings', 'sources (Fr)']


# In[ ]:


confidence_df.columns


# In[ ]:


X = confidence_df[['number of meanings', 'sources (Fr)']].values


# In[ ]:


y = confidence_df['confidence'].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=range(len(y_test)))


# In[ ]:


df.plot(kind='bar')


# ### Test 1: Mean absolute error should be within the 10% range of the mean confidence value.

# In[ ]:


new_conf_df['confidence'].mean() # mean confidence value = 0.6191 >> MAE should be not greater than 0.06


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ### Test 2: R squared is in the interval (0, 1), the greater the better

# In[ ]:


metrics.r2_score(y_test, y_pred)


# In[ ]:


metrics.explained_variance_score(y_test, y_pred) # maximum value is 1 and is desirable


# In[ ]:


metrics.max_error(y_test, y_pred) 


# In[ ]:


metrics.median_absolute_error(y_test, y_pred)


# In[ ]:


# add result to a dictionary
models_results.append({
    'model': model, 
    'predictors': predictors,
    'additional': 'random_state=0',
    'MAE': metrics.mean_absolute_error(y_test, y_pred),
    'MSE': metrics.mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    'r squared': metrics.r2_score(y_test, y_pred),
    'explained variance score': metrics.explained_variance_score(y_test, y_pred),
    'max error': metrics.max_error(y_test, y_pred)
})


# # Model #8: based on (1) number of sources and (2) expected confidence: meanings

# In[ ]:


model = '#8'
predictors = ['expected confidence: nr of meanings', 'sources (Fr)']


# In[ ]:


confidence_df.columns


# In[ ]:


X = confidence_df[['expected confidence: nr of meanings', 'sources (Fr)']].values


# In[ ]:


y = confidence_df['confidence'].values


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


y_test


# In[ ]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}, index=range(len(y_test)))


# In[ ]:


df.plot(kind='bar')


# ### Test 1: Mean absolute error should be within the 10% range of the mean confidence value.

# In[ ]:


confidence_df['confidence'].mean() # mean confidence value = 0.6191 >> MAE should be not greater than 0.06


# In[ ]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ### Test 2: R squared is in the interval (0, 1), the greater the better

# In[ ]:


metrics.r2_score(y_test, y_pred)


# In[ ]:


metrics.explained_variance_score(y_test, y_pred) # maximum value is 1 and is desirable


# In[ ]:


metrics.max_error(y_test, y_pred) 


# In[ ]:


metrics.median_absolute_error(y_test, y_pred)


# In[ ]:


# add result to a dictionary
models_results.append({
    'model': model, 
    'predictors': predictors,
    'additional': 'random_state=42',
    'MAE': metrics.mean_absolute_error(y_test, y_pred),
    'MSE': metrics.mean_squared_error(y_test, y_pred),
    'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    'r squared': metrics.r2_score(y_test, y_pred),
    'explained variance score': metrics.explained_variance_score(y_test, y_pred),
    'max error': metrics.max_error(y_test, y_pred)
})


# ## Model performance comparison

# In[ ]:


pp.pprint(models_results)


# In[ ]:


model_performance = pd.DataFrame(models_results)


# In[ ]:


model_performance


# In[ ]:


model_performance.to_clipboard()


# In[ ]:


annotation_styles.iloc[:,:5].to_clipboard()


# As it seems, the best two models to predict the confidence of a word annotation so far are:
# 
# 1) the model based on the expected confidence for the word with the same number of meanings, the expected confidence of the annotator, and the confidence that the annotator showed when annotating the test word 'virtus'
# 
# 2) the model based on the count of patterns and number of meaning clusters / number of meanings for the word
# 
# It seems that the annotators' personal differences in confidence and 'annotation style' do not contribute as much to the overall confidence of the annotation as the qualities of the word that was annotated. 
