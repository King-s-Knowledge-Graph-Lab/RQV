import pandas as pd
import numpy as np
from ast import literal_eval as leval
import seaborn as sns
from tqdm.auto import tqdm
tqdm.pandas()
from matplotlib import pyplot as plt

def get_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    outliers = series[(series < fence_low) | (series > fence_high)]
    return outliers

""" 
from emoji import UNICODE_EMOJI
def is_emoji(s):
    flags = re.findall(u'[\U0001F1E6-\U0001F1FF]', s)
    if flags:
        return True
    return s in UNICODE_EMOJI
"""

## 1. Get WTR dataset
reference_text_df = pd.read_csv('WD_Textual_References_Dataset/reference_html_as_sentences_df.csv')
reference_text_df.info()
def check_column_dist(df, col):
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, 'counts']
    counts['per'] = 100*counts['counts']/counts['counts'].sum()
    counts[col] = counts[col].astype(str)
    return counts

# Netlocs are equaly spread
# codes are all good
# reasons are good
fig, ax = plt.subplots(1,3,figsize=(15,10))
sns.barplot(data=check_column_dist(reference_text_df, 'netloc_agg'), x='per', y='netloc_agg', ax=ax[0])
sns.barplot(data=check_column_dist(reference_text_df, 'code'), x='per', y='code', ax=ax[1])
sns.barplot(data=check_column_dist(reference_text_df, 'reason'), x='per', y='reason', ax=ax[2])
plt.tight_layout()
plt.show()
claim_data_df = pd.read_csv('WD_Textual_References_Dataset/text_reference_claims_df.csv')
claim_data_df.info()
print('Total counts:')
print(f'{claim_data_df.claim_id.unique().shape[0]} unique claims')
print(f'{claim_data_df.reference_id.unique().shape[0]} unique references')
# Considerable amount of aliases for entities, LOTS for properties, reasonable for objects.
# We can use this to generate multiple verbalisations based on aliases.
fig, ax = plt.subplots(4,3,figsize=(15,10))
sns.barplot(data=check_column_dist(claim_data_df, 'rank'), x='per', y='rank', ax=ax[0][0])
sns.barplot(data=check_column_dist(claim_data_df, 'datatype'), x='per', y='datatype', ax=ax[0][1])
#sns.barplot(data=check_column_dist(claim_data_df, 'entity_label_lan'), x='per', y='entity_label_lan', ax=ax[0][2])
sns.barplot(data=check_column_dist(claim_data_df, 'entity_label_lan'), x='per', y='entity_label_lan', ax=ax[1][0])
sns.barplot(data=check_column_dist(claim_data_df, 'entity_alias_lan'), x='per', y='entity_alias_lan', ax=ax[1][1])
sns.barplot(data=check_column_dist(claim_data_df, 'entity_desc_lan'), x='per', y='entity_desc_lan', ax=ax[1][2])
sns.barplot(data=check_column_dist(claim_data_df, 'property_label_lan'), x='per', y='property_label_lan', ax=ax[2][0])
sns.barplot(data=check_column_dist(claim_data_df, 'property_alias_lan'), x='per', y='property_alias_lan', ax=ax[2][1])
sns.barplot(data=check_column_dist(claim_data_df, 'property_desc_lan'), x='per', y='property_desc_lan', ax=ax[2][2])
sns.barplot(data=check_column_dist(claim_data_df, 'object_label_lan'), x='per', y='object_label_lan', ax=ax[3][0])
sns.barplot(data=check_column_dist(claim_data_df, 'object_alias_lan'), x='per', y='object_alias_lan', ax=ax[3][1])
sns.barplot(data=check_column_dist(claim_data_df, 'object_desc_lan'), x='per', y='object_desc_lan', ax=ax[3][2])
plt.tight_layout()
plt.show()

""" 
# Distribution of entities and properties involved in the claim data
fig, ax = plt.subplots(4,2,figsize=(10,12))

ax[0][0].set_yscale('log')
entity_count = claim_data_df.entity_id.value_counts()
ax[0][0].set_title('Boxplot of Entity distribution')
sns.boxplot(data = entity_count, ax=ax[0][0])
ax[0][1].set_title('KDE of Entity distribution')
sns.kdeplot(data = entity_count, ax=ax[0][1])

ax[1][0].set_yscale('log')
property_count = claim_data_df.property_id.value_counts()
ax[1][0].set_title('Boxplot of Property distribution')
sns.boxplot(data = property_count, ax=ax[1][0])
ax[1][1].set_title('KDE of Property distribution')
sns.kdeplot(data = property_count, ax=ax[1][1])

datatype_count = claim_data_df.datatype.value_counts().reset_index()
ax[2][0].set_title('Barplot of Datatype distribution')
sns.barplot(data = datatype_count, ax=ax[2][0], x='index', y='datatype')

entity_alias_count = claim_data_df.entity_alias.apply(lambda x : len(leval(x)) if x != 'no-alias' else 0)
ax[2][1].set_title('KDE of Entity alias count distribution')
sns.kdeplot(data = entity_alias_count, ax=ax[2][1])

entity_alias_count = claim_data_df.property_alias.apply(lambda x : len(leval(x)) if x != 'no-alias' else 0)
ax[3][0].set_title('KDE of Property alias count distribution')
sns.kdeplot(data = entity_alias_count, ax=ax[3][0])

entity_alias_count = claim_data_df.object_alias.apply(lambda x : len(leval(x)) if x != 'no-alias' else 0)
ax[3][1].set_title('KDE of Object alias count distribution')
sns.kdeplot(data = entity_alias_count, ax=ax[3][1])

plt.tight_layout()
plt.show()
"""
# We can see that entities and properties are somewhat fairly spread, with most within a lesser volume and a few outliers.
# Datatype has an OK distribution, nothing wrong there for this analysis.
# KDE of alias counts for ent, prop, and obj show most cases on low count and a few outliers.

## 2. Verbalisation
from Prove.verbalisation import verbalisation_module

# If updating the module
#from importlib import reload
#reload(verbalisation_module)

verb_module = verbalisation_module.VerbModule()

import torch
torch.cuda.is_available(),\
torch.cuda.device_count(),\
torch.cuda.current_device(),\
torch.cuda.device(0),\
torch.cuda.get_device_name(0)


try:
    verbalised_claims_df = pd.read_csv('verbalisation/verbalised_claims_df.csv')
except Exception:
    verbalised_claims_df = None
verbalised_claims_df

import json
import time

BATCH_SIZE = 16
verbalised_claims_this_batch = []

claim_data_to_keep = [
    'reference_id', 'entity_id', 'claim_id', 'rank', 'property_id', 'datatype',
    'entity_label', 'entity_desc', 'property_label', 'property_desc', 'object_label', 'object_desc'
] # also add entity_label_is_alias, same for property and object

if verbalised_claims_df is not None and not verbalised_claims_df.empty:
    verbalised_claims = json.loads(
        verbalised_claims_df[verbalised_claims_df['verbalisation'] != 'NO_VERBALISATION'].to_json(orient="records")
    )
else:
    verbalised_claims = []

print(len(verbalised_claims))

with open('verbalisation.log','w+',encoding='utf-8') as f:

    for i, row in tqdm(claim_data_df.iterrows(), total=claim_data_df.shape[0]):
    
        try:

            subjects = [row['entity_label']] if row['entity_label_lan'] == 'en' else []
            subjects += leval(row['entity_alias']) if row['entity_alias_lan'] == 'en' else []

            for i_s, subject in enumerate(subjects):

                predicates = [row['property_label']] if row['property_label_lan'] == 'en' else []
                predicates += leval(row['property_alias']) if row['property_alias_lan'] == 'en' else []

                for i_p, predicate in enumerate(predicates):

                    objects = [row['object_label']] if row['object_label_lan'] == 'en' else []
                    objects += leval(row['object_alias']) if row['object_alias_lan'] == 'en' else []

                    for i_o, object_ in enumerate(objects):

                        verbalised_claim_entry = {
                            c : row[c] for c in claim_data_to_keep
                        }                        
                        
                        if type(verbalised_claims_df) == pd.core.frame.DataFrame:
                            if not verbalised_claims_df[
                                (verbalised_claims_df['reference_id'] == row['reference_id']) &\
                                (verbalised_claims_df['claim_id'] == row['claim_id']) &\
                                (verbalised_claims_df['entity_label'] == subject) &\
                                (verbalised_claims_df['object_label'] == object_) &\
                                (verbalised_claims_df['property_label'] == predicate)                        
                            ].empty:
                                continue                        
                        #print('nonempty found:','-'.join([subject, predicate, object_]))

                        verbalised_claim_entry.update({
                            'entity_label_is_alias': (i_s != 0),
                            'property_label_is_alias': (i_p != 0),
                            'object_label_is_alias': (i_o != 0),
                            'entity_label': subject,
                            'property_label': predicate,
                            'object_label': object_
                        })
                        
                        #print(f"[{i}/{claim_data_df.shape[0]-1}, {i_s}/{len(subjects)-1}, {i_p}/{len(predicates)-1}, {i_o}/{len(objects)-1}]")
                        is_last =\
                            (i == claim_data_df.shape[0]-1) &\
                            (i_s == len(subjects)-1) &\
                            (i_p == len(predicates)-1) &\
                            (i_o == len(objects)-1)
                        #if is_last:
                        #    print('LAST')
                            
                        
                        verbalised_claims_this_batch.append(verbalised_claim_entry)
                        if len(verbalised_claims_this_batch) >= BATCH_SIZE or is_last:
                            #print('verbalising...')
                            verbalisation_inputs = [{
                                'subject':e['entity_label'], 'predicate':e['property_label'], 'object':e['object_label']
                            } for e in verbalised_claims_this_batch]

                            try:
                                #verbalisations = verb_module.verbalise(verbalisation_inputs)
                                verbalisations = ['TEST' for _ in verbalisation_inputs]
                                #f.write(
                                #    f'Verbalising: {[(v["reference_id"], v['claim_id']) for v in verbalised_claims_this_batch]}\n'
                                #)
                            except Exception:
                                print('ERROR VERBALISING:', verbalisation_inputs, sep='\n')
                                raise            

                            for j in range(len(verbalised_claims_this_batch)):
                                try:
                                    verb_module.add_label_to_unk_replacer(verbalised_claims_this_batch[j]['entity_label'])
                                    verb_module.add_label_to_unk_replacer(verbalised_claims_this_batch[j]['object_label'])
                                    verbalised_claims_this_batch[j].update({
                                        'verbalisation' : verbalisations[j],
                                        'verbalisation_unks_replaced': verb_module.replace_unks_on_sentence(
                                            verbalisations[j], empty_after=True
                                        )
                                    })
                                except Exception:
                                    print('REPLACING_ERROR on', verbalised_claims_this_batch[j])
                                    verbalised_claims_this_batch[j].update({
                                        'verbalisation' : verbalisations[j],
                                        'verbalisation_unks_replaced': 'REPLACING_ERROR'
                                    })

                            verbalised_claims += verbalised_claims_this_batch
                            verbalised_claims_this_batch = []

        except Exception:
            print(row)
            #pprint(verbalised_claims_this_row)
            raise

# Load 
verbalised_claims_df = pd.DataFrame(verbalised_claims)

import re

# REMOVING REMAINING <UNK> TOKENS
verbalised_claims_df['verbalisation_unks_replaced_then_dropped'] = None
for i, row in tqdm(verbalised_claims_df.iterrows(), total=verbalised_claims_df.shape[0]):
    s = row['verbalisation_unks_replaced']
    # Removing remaining <unk> tokens
    s = re.sub('<unk>', '', s)
    # Removing doublespaces
    s = re.sub(r'\s+', ' ', s).strip()
    # Removing spaces before punctuation
    s = re.sub(r'\s([?.!",](?:\s|$))', r'\1', s)
    
    verbalised_claims_df.loc[i, 'verbalisation_unks_replaced_then_dropped'] = s

    # Checking if every combination of ENGLISH label+alias for (s,p,o) tuples has been covered
# NO OUTPUTS = ALL OK

claims_list = verbalised_claims_df.claim_id.unique().tolist()
for c_id in claims_list:
    row = claim_data_df[claim_data_df['claim_id'] == c_id]
    matching_claims = row.shape[0]
    verbalised_n_rows = verbalised_claims_df[verbalised_claims_df.claim_id == c_id].shape[0]
    
    verbalised_n_rows_target =\
        ((len(leval(row['entity_alias'].values[0])) if row['entity_alias_lan'].iloc[0] == 'en' else 0) +1) * \
        ((len(leval(row['property_alias'].values[0])) if row['property_alias_lan'].iloc[0] == 'en' else 0) +1) * \
        ((len(leval(row['object_alias'].values[0])) if row['object_alias_lan'].iloc[0] == 'en' else 0) +1)
    
    try:
        assert verbalised_n_rows == verbalised_n_rows_target * matching_claims
    except AssertionError:
        print('Match Error:',c_id)
        print( row.index.values,
            verbalised_n_rows, 'out of', verbalised_n_rows_target * matching_claims,
            f'{verbalised_n_rows_target * matching_claims - verbalised_n_rows} to go')

verbalised_claims_df.to_csv('verbalisation/verbalised_claims_df.csv', index=None)

#### Verbalisation data correction
verbalised_claims_df = pd.read_csv('verbalisation/verbalised_claims_df.csv')
verbalised_claims_df.info()
verbalised_claims_df.head()
# Define property aliases to use as main verbalisations
special_properties = {
    'P1031': 'citation',
    'P106': 'profession',
    'P1066': 'apprentice of',
    'P1196': 'nature of death',
    'P1308': 'position holder',
    'P131': 'is located in',
    'P1346': 'won by',
    'P136': {#dict replaces if the key is either in the entity label or the object label
        'film': 'film genre', # Manually replace here, as 'genre of' would be good but is not there
        '': 'genre' #default
    },
    'P1435': 'designation', # Manually replace the Overhailes case
    'P1441': 'featured in work',
    'P1448': 'name',
    'P1476': 'titled',
    'P1542': 'causes',
    'P1559': 'native name',
    'P166' : {
        'Doctor': 'recognition title',
        '': 'award received'
    },
    'P17': {
        'trial': 'host country',
        '': 'land'# Manually correct cases where food is introduced (Gups Ponmala)
    },
    'P186': 'made from',
    'P189': 'found in',
    'P195': 'art collection',#Manually correct cases due to extensive entity label formatting
    'P2017': 'isomeric SMILES', #this is the main label, this is just to remind myself to manually correct them due to extensive label format
    'P21': 'gender',
    'P233': 'SMILES', #Manually correct cases due to extensive entity label formatting
    'P26': 'marry', #Manually correct some cases due to tense
    'P279': 'is a type of',
    'P2896': 'publication frequency',#Manually correct cases due to missing link between 1 week = weekly expressions.
    'P31': 'is a',
    'P3373': 'is sibling of',
    'P364': 'original language',
    'P39' : 'held position',
    'P40' : 'has child',
    'P451' : 'is partner of',
    'P452' : 'sector',#Manually correct line of credit cases
    'P485': 'archive location',
    'P5021': 'assessment', #Manually correct inversion cases
    'P527': 'parts',
    'P551': 'resided in',
    'P571': 'created',
    'P580': 'starting',
    'P582': 'ending',
    'P607': 'in conflict',
    'P674': 'characters', #Manually correct cases here
    'P725': 'voice actor', #Manually correct cases here
    'P734': 'last name',
    'P735': 'first name',
    'P780': 'symptoms',
    'P793': 'event',
    'P802': 'students',
    #'P8045' Manually correct this one
    'P915': 'filmed at',
    'P921': 'about',
    'P97': 'hereditary title'
}

verbalised_claims_df['is_main_verbalisation'] = None
verbalised_claims_df['alternative_alias_used'] = None

for i, row in verbalised_claims_df.iterrows():
    # If a special property (to be replaced by alias), do a custom logic where
    # an official property_label is elected instead
    if row['property_id'] in special_properties.keys():   
        verbalised_claims_df.loc[i, 'alternative_alias_used'] = True
        preferred_property_label = special_properties[row['property_id']]
        if type(preferred_property_label) == str:
            # if the new label is str, just flag as main verbalisation in case the entity/object are main labels
            # and the property label is the new official label
            if not row['entity_label_is_alias'] and\
                row['property_label'] == preferred_property_label and\
                not row['object_label_is_alias']:
                
                verbalised_claims_df.loc[i, 'is_main_verbalisation'] = True
            else:
                verbalised_claims_df.loc[i, 'is_main_verbalisation'] = False
        elif type(preferred_property_label) == dict:
            # if it's a dict, elect as new official label only if either entity or object labels contain
            # the key, and select the property label that is its value
            # if one is identified this way, skip the rest
            # we start with it not being a main verbalisation until we find a key that matches
            verbalised_claims_df.loc[i, 'is_main_verbalisation'] = False
            
            # check if another row with the same claim id and reference id is 
            # not already the main verbalisation
            if verbalised_claims_df[
                (verbalised_claims_df['claim_id'] == row['claim_id']) &\
                (verbalised_claims_df['reference_id'] == row['reference_id'])
            ].is_main_verbalisation.sum() > 0:
                continue
                
            for key in preferred_property_label.keys():
                if key in row['entity_label'] or key in row['object_label']:
                    if not row['entity_label_is_alias'] and\
                        row['property_label'] == preferred_property_label[key] and\
                        not row['object_label_is_alias']:
                        #print(row)
                        verbalised_claims_df.loc[i, 'is_main_verbalisation'] = True
                        if key=='':                            
                            verbalised_claims_df.loc[i, 'alternative_alias_used'] = False
                        break #stop iterating over keys of preferred_property_label
                
                
    # Else, or if we fail to find an official property_label through the logic above,
    # just check if all X_is_alias columns are false
    else:        
        verbalised_claims_df.loc[i, 'alternative_alias_used'] = False
        if not row['entity_label_is_alias'] and\
            not row['property_label_is_alias'] and\
            not row['object_label_is_alias']:

            verbalised_claims_df.loc[i, 'is_main_verbalisation'] = True
        else:
            verbalised_claims_df.loc[i, 'is_main_verbalisation'] = False
#assert verbalised_claims_df['is_main_verbalisation'].value_counts()[True] == 972
#972 is the total of unique ref_claim pairs
assert verbalised_claims_df['is_main_verbalisation'].isna().sum() == 0
assert verbalised_claims_df['alternative_alias_used'].isna().sum() == 0
# no empty cells
verbalised_claims_df[verbalised_claims_df['is_main_verbalisation'] == True][
    verbalised_claims_df[verbalised_claims_df['is_main_verbalisation'] == True].duplicated('claim_id')
]
print('Total verbalised counts:')
print(f'{verbalised_claims_df.claim_id.unique().shape[0]} unique claims')
print(f'{verbalised_claims_df.reference_id.unique().shape[0]} unique references')
# 
n_unk_replacements = verbalised_claims_df[verbalised_claims_df['verbalisation'].apply(lambda x : '<unk>' in x)].shape[0]
print(
    f"Unk replacement was needed in {n_unk_replacements} ({100*n_unk_replacements/verbalised_claims_df.shape[0]}%) of verbalisations"
)

n_unk_replacements_solved = verbalised_claims_df[verbalised_claims_df['verbalisation_unks_replaced'].apply(lambda x : '<unk>' in x)].shape[0]
print(
    f"Unk replacement was NOT solved in {n_unk_replacements_solved} ({100*n_unk_replacements_solved/(n_unk_replacements+0.1)}%) of cases"
)

unique_verbalisation_counts = verbalised_claims_df[['claim_id', 'verbalisation_unks_replaced_then_dropped']].\
    drop_duplicates().claim_id.value_counts()

# Distribution of entities and properties involved in the verbalisations

fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].set_yscale('log')
ax[0].set_title('Boxplot of Unique verbalisation count\n distribution per unique claim id')
sns.boxplot(data = unique_verbalisation_counts, ax=ax[0])
ax[1].set_title('KDE of Unique verbalisation count\n distribution per unique claim id')
sns.kdeplot(data = unique_verbalisation_counts, ax=ax[1])

plt.tight_layout()
plt.show()

# This shows the majority of claims have up to 100 unique verbalisations due to aliases, with some having way more.
# The biggest one, for instance, has 33 subject aliases, 22 predicate aliases, and 8 object aliases

get_outliers(unique_verbalisation_counts).reset_index().rename({'index':'claim_id','claim_id':'count'}, axis=1)

# Next step is manually correcting some verbalisations before SENTENCE SELECTION
#Create a 'corrected_verbalisation' and a 'is_corrected_verbalisation' for manual annotation before sentence selection
verbalised_claims_df_main = verbalised_claims_df[verbalised_claims_df['is_main_verbalisation'] == True]\
    .reset_index(drop=True).drop('is_main_verbalisation', axis=1).copy()

verbalised_claims_df_main['corrected_verbalisation'] = verbalised_claims_df_main['verbalisation_unks_replaced_then_dropped']
verbalised_claims_df_main['is_corrected_verbalisation'] = False

verbalised_claims_df_main.to_csv('verbalisation/verbalised_claims_df_main.csv', index=None)

print('Percentage of main verbalisations where an alias was used for better verbalisation')
print(100*verbalised_claims_df_main['alternative_alias_used'].sum()/verbalised_claims_df_main.shape[0])
# Percentage of main verbalisations where an alias was used for better verbalisation

from Levenshtein import distance as levenshtein_distance

verbalised_claims_df_main = pd.read_csv('verbalisation/verbalised_claims_df_main.csv')
verbalised_claims_df_main_corrected = pd.read_csv('verbalisation/verbalised_claims_df_main_corrected.csv')

# is_corrected_verbalisation has NOT been filled during correction, as this is quicker and less error-prone
verbalised_claims_df_main_corrected['is_corrected_verbalisation'] = verbalised_claims_df_main_corrected.apply(
    lambda row : row['corrected_verbalisation'] != row['verbalisation_unks_replaced_then_dropped'], axis=1
)

print('Percentage of main verbalisations where a manual correction was used.')
print(100*verbalised_claims_df_main_corrected['is_corrected_verbalisation'].sum()/verbalised_claims_df_main.shape[0])

norm_levenshtein_distances = verbalised_claims_df_main_corrected.apply(
    lambda row : levenshtein_distance(
        row['corrected_verbalisation'],
        row['verbalisation_unks_replaced_then_dropped']
    )/max(
        len(row['corrected_verbalisation']),
        len(row['verbalisation_unks_replaced_then_dropped'])
    ),
    axis=1
)
norm_levenshtein_distances = norm_levenshtein_distances[norm_levenshtein_distances>0].reset_index(drop=True)
sns.boxplot(data=norm_levenshtein_distances, orient='h')
print('Distribution of normalised levenshtein distance after corrections.')
print(norm_levenshtein_distances.describe())

# REMOVE P1448 (OFFICIAL NAME), P1476 (TITLE), AND P1889 (DIFFERENT) AS THEY ARE REDUNDANT AND NON-INFORMATIVE
#also look at the dataset creation for other properties that were deleted and delete them too

BAD_PROPERTIES = [
    'P1448', # offical name
    'P1476', # title
    'P1889',# different
    'P31', # - instance of
    'P279',# - subclass of
    'P373',# - commons category
    'P910',# - Topic's main category
    'P7561',# - category for the interior of the item
    'P5008',# - on focus list of Wikimedia project
    'P2670',# -  has parts of the class
    'P1740',# -  category for films shot at this location
    'P1612',# -  Commons Institution page
    'P8989',# -  category for the view of the item
    'P2959',# -  permanent duplicated item
    'P7867',# -  category for maps
    'P935' ,# -  Commons gallery
    'P1472',#  -  Commons Creator page
    'P8596',# category for the exterior of the item
    'P5105',# Deutsche Bahn station category
    'P8933',# category for the view from the item
    'P642',# of
    'P3876',# category for alumni of educational institution
    'P1791',# category of people buried here
    'P7084',# related category
    'P1465',# category for people who died here
    'P1687',# Wikidata property
    'P6104',# maintained by WikiProject
    'P4195',# category for employees of the organization
    'P1792',# category of associated people
    'P5869',# model item
    'P1659',# see also
    'P1464',# category for people born here
    'P2354',# has list
    'P1424',# topic's main template
    'P7782',# category for ship name
    'P179',# part of the series
    'P7888',# merged into
    'P6365',# member category
    'P8464',# content partnership category
    'P360',# is a list of
    'P805',# statement is subject of
    'P8703',# entry in abbreviations table
    'P1456',# list of monuments
    'P1012',# including
    'P1151',# topic's main Wikimedia portal
    'P2490',# page at OSTIS Belarus Wiki
    'P593',# HomoloGene ID
    'P8744',# economy of topic
    'P2614',# World Heritage criteria
    'P2184',# history of topic
    'P9241',# demographics of topic
    'P487',#Unicode character
    'P1754',#category related to list
    'P2559',#Wikidata usage instructions
    'P2517',#category for recipients of this award
    'P971',#category combines topics
    'P6112',# category for members of a team
    'P4224',#category contains
    'P301',#category's main topic
    'P1753',#list related to category
    'P1423',#template has topic
    'P1204',#Wikimedia portal's main topic
    'P3921',#Wikidata SPARQL query equivalent
    'P1963',#properties for this type
    'P5125',#Wikimedia outline
    'P3176',#uses property
    'P8952',#inappropriate property for this type
    'P2306',#property
    'P5193',#Wikidata property example for forms
    'P5977',#Wikidata property example for senses
    'P1748',#NCI Thesaurus ID
    'P1692',#ICD-9-CM
    'P248',#stated in
]

verbalised_claims_df_main_corrected_badpropdrop = verbalised_claims_df_main_corrected[
    ~verbalised_claims_df_main_corrected['property_id'].isin(BAD_PROPERTIES)
]
print('Percentage [Number] of claims dropped due to bad properties')
print(
    f'{100 - 100*verbalised_claims_df_main_corrected_badpropdrop.shape[0]/verbalised_claims_df_main_corrected.shape[0]}',
    f'[{verbalised_claims_df_main_corrected.shape[0] - verbalised_claims_df_main_corrected_badpropdrop.shape[0]}]'
)

# Correct some final_urls in the reference_text_df dataframe before joining
import re

# Replace by archived page if page was behing paywall when parsed
reference_text_df.loc[reference_text_df.html.map(lambda x : '://archive.ph/' in x), 'final_url'] =\
    reference_text_df.loc[reference_text_df.html.map(lambda x : '://archive.ph/' in x)]\
        .html.map(lambda x : re.findall(r'http(?:s){0,1}://archive.ph/(?:[a-zA-Z0-9]*)', x)[0])

        # Get URLs from the references df
verbalised_claims_df_main_corrected_badpropdrop_url = \
    pd.merge(
    verbalised_claims_df_main_corrected_badpropdrop,
    reference_text_df[['reference_id', 'final_url']],
    on='reference_id'
)
# Remove duplicates of reference and verbalisation, as duplicates arise from qualifier dependancy
verbalised_claims_df_main_corrected_badpropdrop_url_duplidrop = \
    verbalised_claims_df_main_corrected_badpropdrop_url.drop_duplicates(
    ['corrected_verbalisation','final_url'], keep='first'
)

print('Percentage [Number] of claims dropped due to duplicated verbalisation and url pair')
print(
    f'{100 - 100*verbalised_claims_df_main_corrected_badpropdrop_url_duplidrop.shape[0]/verbalised_claims_df_main_corrected_badpropdrop_url.shape[0]}',
    f'[{verbalised_claims_df_main_corrected_badpropdrop_url.shape[0] - verbalised_claims_df_main_corrected_badpropdrop_url_duplidrop.shape[0]}]'
)
# Remove the three cases in archinform.net written in German
verbalised_claims_df_main_corrected_badpropdrop_url_duplidrop = verbalised_claims_df_main_corrected_badpropdrop_url_duplidrop[
    ~verbalised_claims_df_main_corrected_badpropdrop_url_duplidrop['final_url'].map(
        lambda x : 'www.archinform.net' in x and any([(y in x) for y in ['19632','11996','45859']])
    )
]
verbalised_claims_df_main_corrected_badpropdrop_url_duplidrop.reset_index(drop=True, inplace=True)
verbalised_claims_df_main_corrected_badpropdrop_url_duplidrop
# The following claims were selected randomly such that
# each reference_id had only one claim selected.
# However, we did not keep the seed which generated it.
randomly_selected_rows = [
    0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 18, 22, 23, 27, 29, 32, 36, 39, 43, 44, 49,
    51, 58, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 80, 81, 82, 83, 84, 85, 86,
    87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115, 116, 118, 120, 122, 123, 125, 126, 128, 130, 131,
    132, 133, 135, 136, 140, 141, 142, 147, 151, 152, 153, 154, 155, 158, 159, 160, 161, 163,
    164, 166, 168, 169, 170, 171, 174, 175, 176, 177, 178, 185, 186, 188, 190, 193, 194, 197,
    198, 200, 202, 204, 206, 207, 211, 212, 214, 217, 219, 220, 222, 224, 229, 230, 231, 233,
    235, 239, 240, 241, 243, 244, 246, 247, 250, 251, 253, 254, 255, 256, 257, 258, 259, 260,
    261, 262, 263, 264, 265, 266, 268, 269, 270, 271, 272, 274, 276, 277, 280, 281, 283, 285,
    286, 288, 290, 292, 294, 299, 300, 301, 302, 303, 312, 319, 323, 324, 325, 326, 327, 328,
    331, 332, 335, 336, 338, 340, 342, 344, 345, 347, 350, 352, 353, 354, 355, 356, 357, 358,
    359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 371, 372, 373, 375, 376, 377, 378,
    379, 380, 382, 389, 391, 399, 407, 409, 414, 415, 416, 418, 419, 420, 422, 423, 424, 425,
    426, 427, 428, 429, 432, 439, 442, 443, 445, 446, 448, 454, 456, 458, 461, 464, 465, 467,
    468, 470, 472, 475, 476, 478, 481, 483, 484, 486, 488, 491, 492, 496, 499, 503, 508, 509,
    516, 519, 524, 526, 528, 536, 537, 539, 540, 542, 544, 545, 546, 547, 548, 551, 552, 557,
    561, 570, 571, 573, 576, 578, 579, 581, 583, 584, 586, 588, 589, 590, 592, 593, 594, 595,
    596, 597, 598, 599, 600, 601, 604, 606, 607, 613, 614, 615, 616, 617, 619, 620, 621, 629,
    635, 642, 644, 645, 646, 650, 656, 665, 666, 673, 681, 687, 694, 696, 707, 713, 717, 721,
    729, 730, 731, 732, 736, 737, 738, 739, 740, 741, 743, 744, 745, 747, 748, 749, 750, 751,
    752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 764, 766, 768, 769, 772, 773, 774,
    775, 776, 777, 779, 780, 781, 782, 784, 785, 786, 787, 790, 791, 792, 793, 795
]
verbalised_claims_df_final = verbalised_claims_df_main_corrected_badpropdrop_url_duplidrop
verbalised_claims_df_final = verbalised_claims_df_final.loc[randomly_selected_rows]
verbalised_claims_df_final = verbalised_claims_df_final.reset_index(drop=True)
verbalised_claims_df_final.to_csv('verbalisation/verbalised_claims_df_final.csv', index=None)
verbalised_claims_df_final

## 3. Sentence Selection
verbalised_claims_df_final = pd.read_csv('verbalisation/verbalised_claims_df_final.csv')
verbalised_claims_df_final = verbalised_claims_df_final
verbalised_claims_df_final.info()
# Remove redundant columns and unnecessary columns for the merging with reference contents data
verbalised_claims_df_final = verbalised_claims_df_final[[
    'reference_id', 'claim_id', 'corrected_verbalisation'
]]
verbalised_claims_df_final = verbalised_claims_df_final.rename(
    {'corrected_verbalisation': 'final_verbalisation'},
    axis=1
)


# Take only the data relevant for joining with the verbalisation data and calculating sentence relevance scores

# The sampling weights CAN ONLY BE USED to average-out any score or quantifiable property that is assigned to EACH REFERENCE,
# such as: percentage of claims actually supported by the reference out of all claims linked to it according
# to pipeline results

reference_text_df_for_sentence_selection_join = reference_text_df[[
    'reference_id', 'sampling_weight', 'final_url', 'netloc_agg', 'nlp_sentences', 'nlp_sentences_slide_2'
]]


sentence_relevance_df = pd.merge(
    verbalised_claims_df_final,
    reference_text_df_for_sentence_selection_join,
    how='left',
    on='reference_id'
)

sentence_relevance_df['nlp_sentences'] = sentence_relevance_df['nlp_sentences'].apply(leval)
sentence_relevance_df['nlp_sentences_slide_2'] = sentence_relevance_df['nlp_sentences_slide_2'].apply(leval)

from Prove.SentenceRetrieval import sentence_retrieval_module

# If updating the module
#from importlib import reload
#reload(sentence_retrieval_module)

sr_module = sentence_retrieval_module.SentenceRetrievalModule(max_len=512)

import pdb
BATCH_SIZE = 16

sentence_relevance_df['nlp_sentences_scores'] = None
sentence_relevance_df['nlp_sentences_slide_2_scores'] = None

def chunks(l, n):
    n = max(1, n)
    return [l[i:i+n] for i in range(0, len(l), n)]

all_outputs = []
for i, row in tqdm(sentence_relevance_df.iterrows(), total=sentence_relevance_df.shape[0]):
    
    outputs = []
    for batch in chunks(row['nlp_sentences'], BATCH_SIZE):
        batch_outputs = sr_module.score_sentence_pairs(
            [(row['final_verbalisation'], sentence) for sentence in batch]
        )
        outputs += batch_outputs
    all_outputs.append(outputs)
    
all_outputs = pd.Series(all_outputs)
sentence_relevance_df['nlp_sentences_scores'] = all_outputs

assert all(sentence_relevance_df.apply(
    lambda x : len(x['nlp_sentences']) == len(x['nlp_sentences_scores']),
    axis=1
))

all_outputs = []
for i, row in tqdm(sentence_relevance_df.iterrows(), total=sentence_relevance_df.shape[0]):
    
    outputs = []
    for batch in chunks(row['nlp_sentences_slide_2'], BATCH_SIZE):
        batch_outputs = sr_module.score_sentence_pairs(
            [(row['final_verbalisation'], sentence) for sentence in batch]
        )
        outputs += batch_outputs
    all_outputs.append(outputs)
    
all_outputs = pd.Series(all_outputs)    
sentence_relevance_df['nlp_sentences_slide_2_scores'] = all_outputs
    

assert all(sentence_relevance_df.apply(
    lambda x : len(x['nlp_sentences_slide_2']) == len(x['nlp_sentences_slide_2_scores']),
    axis=1
))

N_TOP_SENTENCES = 5
SCORE_THRESHOLD = 0

nlp_sentences_TOP_N = []
nlp_sentences_slide_2_TOP_N = []
nlp_sentences_all_TOP_N = []

for i, row in tqdm(sentence_relevance_df.iterrows(), total=sentence_relevance_df.shape[0]):

    nlp_sentences_with_scores = [{
        'sentence': t[0],
        'score': t[1],
        'sentence_id': str(j)
    } for j, t in enumerate(zip(row['nlp_sentences'], row['nlp_sentences_scores']))]

    nlp_sentences_with_scores = sorted(nlp_sentences_with_scores, key = lambda x : x['score'], reverse = True)
    nlp_sentences_TOP_N.append(nlp_sentences_with_scores[:N_TOP_SENTENCES])
    
    nlp_sentences_slide_2_with_scores = [{
        'sentence': t[0],
        'score': t[1],
        'sentence_id': str(j)+';'+str(j+1)
    } for j, t in enumerate(zip(row['nlp_sentences_slide_2'], row['nlp_sentences_slide_2_scores']))]

    nlp_sentences_slide_2_with_scores = sorted(nlp_sentences_slide_2_with_scores, key = lambda x : x['score'], reverse = True)
    nlp_sentences_slide_2_TOP_N.append(nlp_sentences_slide_2_with_scores[:N_TOP_SENTENCES])
    

    nlp_sentences_all_with_scores = nlp_sentences_with_scores + nlp_sentences_slide_2_with_scores
    nlp_sentences_all_with_scores = sorted(nlp_sentences_all_with_scores, key = lambda x : x['score'], reverse = True)
    
    #We might no want to allow overlaps, so we do the following:
    #For each evidence in descending order of score, we delete from the 'all' list
    #all overlapping evidence scored lower than it
    nlp_sentences_all_with_scores_filtered_for_overlap = []
    for evidence in nlp_sentences_all_with_scores:
        if ';' in evidence['sentence_id']:
            [start_id, end_id] = evidence['sentence_id'].split(';')
            if not any(
                [start_id in e['sentence_id'].split(';') for e in nlp_sentences_all_with_scores_filtered_for_overlap]
            ):
                if not any(
                    [end_id in e['sentence_id'].split(';') for e in nlp_sentences_all_with_scores_filtered_for_overlap]
                ):
                    nlp_sentences_all_with_scores_filtered_for_overlap.append(evidence)
        else:
            if not any(
                [evidence['sentence_id'] in e['sentence_id'].split(';') for e in nlp_sentences_all_with_scores_filtered_for_overlap]
            ):
                nlp_sentences_all_with_scores_filtered_for_overlap.append(evidence)
    
    
    assert len(nlp_sentences_all_with_scores_filtered_for_overlap) >= 5    
    nlp_sentences_all_TOP_N.append(nlp_sentences_all_with_scores_filtered_for_overlap[:N_TOP_SENTENCES])
    
sentence_relevance_df['nlp_sentences_TOP_N'] = pd.Series(nlp_sentences_TOP_N)
sentence_relevance_df['nlp_sentences_slide_2_TOP_N'] = pd.Series(nlp_sentences_slide_2_TOP_N)
sentence_relevance_df['nlp_sentences_all_TOP_N'] = pd.Series(nlp_sentences_all_TOP_N)

sentence_relevance_df.iloc[1].nlp_sentences_all_TOP_N

sentence_relevance_df.to_json(
    'sentence_retrieval/sentence_relevance_df.json', orient='records', indent=4
)

sentence_relevance_df = pd.read_json('sentence_retrieval/sentence_relevance_df.json')
sentence_relevance_df

for netloc in sentence_relevance_df.netloc_agg.unique().tolist():
    print('-'*10)
    print(netloc,'\n')
    sampled_rows = sentence_relevance_df[sentence_relevance_df['netloc_agg'] == netloc].sample(3, random_state=42)
    for i, row in sampled_rows.iterrows():
        print('->', row['final_verbalisation'])
        print('->>', row['final_url'])
        for i, sentence in enumerate(row['nlp_sentences_all_TOP_N']):
            print(f"\t{i+1}. {sentence['sentence']} ({sentence['score']})")
        print()


#4. Textual Entailment
from Prove.textual_entailment import textual_entailment_module
from importlib import reload
reload(textual_entailment_module)

te_module = textual_entailment_module.TextualEntailmentModule()
textual_entailment_df = sentence_relevance_df.copy()

SCORE_THRESHOLD = 0

keys = ['TOP_N', 'slide_2_TOP_N', 'all_TOP_N']
te_columns = {}

for key in keys:
    te_columns[f'evidence_TE_prob_{key}'] = []
    te_columns[f'evidence_TE_prob_weighted_{key}'] = []
    te_columns[f'evidence_TE_labels_{key}'] = []
    te_columns[f'claim_TE_prob_weighted_sum_{key}'] = []
    te_columns[f'claim_TE_label_weighted_sum_{key}'] = []
    te_columns[f'claim_TE_label_malon_{key}'] = []


for i, row in tqdm(textual_entailment_df.iterrows(), total=textual_entailment_df.shape[0]):
    try:
        claim = row['final_verbalisation']

        result_sets = {key : {'evidence': row[f'nlp_sentences_{key}']} for key in keys}

        for key, rs in result_sets.items():

            evidence_size = len([e for e in rs['evidence']])
           
            rs['evidence_TE_prob'] = te_module.get_batch_scores(
                claims = [claim for _ in range(evidence_size)],
                evidence = [e['sentence'] for e in rs['evidence']]
            )   
            
            rs['evidence_TE_labels'] = [te_module.get_label_from_scores(s) for s in rs['evidence_TE_prob']]
                
            rs['evidence_TE_prob_weighted'] = [
                probs*ev['score'] for probs, ev in zip(rs['evidence_TE_prob'], rs['evidence'])\
                if ev['score'] > SCORE_THRESHOLD
            ]
            
            rs['claim_TE_prob_weighted_sum'] = \
                np.sum(rs['evidence_TE_prob_weighted'], axis=0)\
                if rs['evidence_TE_prob_weighted'] else [0,0,0]
            
            rs['claim_TE_label_weighted_sum'] = \
                te_module.get_label_from_scores(rs['claim_TE_prob_weighted_sum'])\
                if rs['evidence_TE_prob_weighted'] else 'NOT ENOUGH INFO'  
            

            rs['claim_TE_label_malon'] = te_module.get_label_malon(
                probs for probs, ev in zip(rs['evidence_TE_prob'], rs['evidence'])\
                if ev['score'] > SCORE_THRESHOLD
            )

            te_columns[f'evidence_TE_prob_{key}'].append(rs['evidence_TE_prob'])
            te_columns[f'evidence_TE_prob_weighted_{key}'].append(rs['evidence_TE_prob_weighted'])
            te_columns[f'evidence_TE_labels_{key}'].append(rs['evidence_TE_labels'])
            te_columns[f'claim_TE_prob_weighted_sum_{key}'].append(rs['claim_TE_prob_weighted_sum'])
            te_columns[f'claim_TE_label_weighted_sum_{key}'].append(rs['claim_TE_label_weighted_sum'])
            te_columns[f'claim_TE_label_malon_{key}'].append(rs['claim_TE_label_malon'])
            
            #print(rs)
            #break
    
    except Exception:
        print(row)
        print(result_sets)

        raise
    
    #break

for key in keys:
    textual_entailment_df[f'evidence_TE_prob_{key}'] = pd.Series(te_columns[f'evidence_TE_prob_{key}'])
    textual_entailment_df[f'evidence_TE_prob_weighted_{key}'] = pd.Series(te_columns[f'evidence_TE_prob_weighted_{key}'])
    textual_entailment_df[f'evidence_TE_labels_{key}'] = pd.Series(te_columns[f'evidence_TE_labels_{key}'])
    textual_entailment_df[f'claim_TE_prob_weighted_sum_{key}'] = pd.Series(te_columns[f'claim_TE_prob_weighted_sum_{key}'])
    textual_entailment_df[f'claim_TE_label_weighted_sum_{key}'] = pd.Series(te_columns[f'claim_TE_label_weighted_sum_{key}'])
    textual_entailment_df[f'claim_TE_label_malon_{key}'] = pd.Series(te_columns[f'claim_TE_label_malon_{key}'])

textual_entailment_df
textual_entailment_df.to_json('Prove/textual_entailment/textual_entailment_df.json', orient="records", indent=4)
textual_entailment_df = pd.read_json('Prove/textual_entailment/textual_entailment_df.json')
SCORE_THRESHOLD = 0
textual_entailment_df.shape
textual_entailment_df.iloc[0]

df = textual_entailment_df.copy()

try:
    for netloc in df.netloc_agg.unique().tolist():
        print('-'*10)
        print(netloc,'\n')
        sampled_rows = df[df['netloc_agg'] == netloc].sample(3, random_state=42)
        for i, row in sampled_rows.iterrows():
            print(
                '->', row['final_verbalisation'],'\n',
                '\t-WS:', row['claim_TE_label_weighted_sum_all_TOP_N'], f"({row['claim_TE_prob_weighted_sum_all_TOP_N']})\n",
                '\t-M: ', row['claim_TE_label_malon_all_TOP_N']
            )
            print('->>', row['final_url'])
            for i, sentence in enumerate(row['nlp_sentences_all_TOP_N']):
                if sentence['score'] > SCORE_THRESHOLD:
                    print(
                        f"\t{i+1}. {sentence['sentence']}\n",
                        f"\t-Evidence Score: {sentence['score']}\n",
                        f"\t-Label Prob: {row['evidence_TE_prob_all_TOP_N'][i]}\n",
                        f"\t-Label Prob Weighted: {row['evidence_TE_prob_weighted_all_TOP_N'][i]}\n",
                        f"\t-Label: {row['evidence_TE_labels_all_TOP_N'][i]}\n",
                    )
                else:
                    print(
                        f"\t{i+1}. {sentence['sentence']}\n",
                        f"\t-Evidence Score: {sentence['score']}\n",
                        f"\t-Label Prob: {row['evidence_TE_prob_all_TOP_N'][i]}\n",
                        f"\t-Label Prob Weighted: {[0,0,0]}\n",
                        f"\t-Label: {row['evidence_TE_labels_all_TOP_N'][i]}\n",
                    )
            print()
except Exception:
    print(row)
    raise