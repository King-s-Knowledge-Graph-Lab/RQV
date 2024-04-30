import numpy as np
from tqdm import tqdm
import pandas as pd
import os, sqlite3, traceback, ast, requests, fasttext, re, time
from requests.exceptions import ReadTimeout, TooManyRedirects, ConnectionError, ConnectTimeout, InvalidSchema, InvalidURL
from qwikidata.linked_data_interface import get_entity_dict_from_api
from datetime import datetime
import WDV.WikidataClaims.wikidata_utils as wdutils
from importlib import reload  
from urllib.parse import urlparse, unquote
from urllib import parse
from bs4 import BeautifulSoup
from IPython.display import clear_output

class DatabaseExtractor():
    def __init__(self, dbname='wikidata_claims_refs_parsed.db'):
        self.dbname = dbname
        if os.path.exists(self.dbname):  # .db initializing
            os.remove(self.dbname) 
        self.prepare_extraction()
        
    def finish_extraction(self):
        self.db.commit()
        
    def prepare_extraction(self):
        self.db = sqlite3.connect(self.dbname)
        self.cursor = self.db.cursor()

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS claims(
                entity_id TEXT,
                claim_id TEXT,
                claim_rank TEXT,
                property_id TEXT,
                datatype TEXT,
                datavalue TEXT,
                PRIMARY KEY (
                    claim_id
                )
        )''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS claims_refs(
                claim_id TEXT,
                reference_id TEXT,
                PRIMARY KEY (
                    claim_id,
                    reference_id
                )
        )''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS refs(
                reference_id TEXT,
                reference_property_id TEXT,
                reference_index TEXT,
                reference_datatype TEXT,
                reference_value TEXT,
                PRIMARY KEY (
                    reference_id,
                    reference_property_id,
                    reference_index
                )
        )''')
        self.db.commit()  
        
    def extract_claim(self, entity_id, claim):
        if claim['mainsnak']['snaktype'] == 'value':
            value = str(claim['mainsnak']['datavalue'])
        else:
            value = claim['mainsnak']['snaktype']
        try:
            self.cursor.execute('''
            INSERT INTO claims(entity_id, claim_id, claim_rank, property_id, datatype, datavalue)
            VALUES($var,$var,$var,$var,$var,$var)'''.replace('$var','?'), (
                entity_id,claim['id'],claim['rank'],
                claim['mainsnak']['property'],claim['mainsnak']['datatype'],value
            ))
        except UnicodeEncodeError:
            print(entity_id,claim['id'],claim['rank'],
                claim['mainsnak']['property'],claim['mainsnak']['datatype'],value)
            raise
        except sqlite3.IntegrityError as err:
            #self.db.rollback()
            self.cursor.execute(
                '''SELECT *
                FROM claims 
                WHERE claim_id=$var
                '''.replace('$var','?'), (claim['id'],)
            )
            conflicted_value = self.cursor.fetchone()
            if conflicted_value == (entity_id,claim['id'],claim['rank'],
                    claim['mainsnak']['property'],claim['mainsnak']['datatype'],value):
                pass
            else:
                print(err, claim['id'])
                traceback.print_exc()
                raise err
        finally:
            #self.db.commit()
            pass

    def extract_reference(self, ref):
        for snaks in ref['snaks'].values():
            for i, snak in enumerate(snaks):
                if snak['snaktype'] == 'value':
                    value = str(snak['datavalue'])
                else:
                    value = snak['snaktype']
                try:
                    self.cursor.execute('''
                    INSERT INTO refs(reference_id, reference_property_id, reference_index,
                    reference_datatype, reference_value)
                    VALUES($var,$var,$var,$var,$var)'''.replace('$var','?'), (
                        ref['hash'],snak['property'],str(i),snak['datatype'],value
                    ))
                except sqlite3.IntegrityError as err:
                    #self.db.rollback()
                    self.cursor.execute(# WE DONT USE THE INDEX HERE, THEY TEND TO COME SHUFFLED FROM API AND SORTING TAKES TOO LONG
                        '''SELECT reference_id, reference_property_id, reference_datatype, reference_value
                        FROM refs 
                        WHERE reference_id = $var
                        AND reference_property_id = $var
                        '''.replace('$var','?'), (ref['hash'],snak['property'])
                    )
                    conflicted_values = self.cursor.fetchall()
                    if  (ref['hash'],snak['property'],snak['datatype'],value) in conflicted_values:
                        pass
                    else:
                        print(err, ref['hash'],snak['property'],i)
                        print('trying to insert:',(ref['hash'],snak['property'],str(i),snak['datatype'],value))
                        traceback.print_exc()
                        raise err
                finally:
                    #self.db.commit()
                    pass
            
    def extract_claim_reference(self, claim, ref):
        claim['id'],ref['hash']
        try:
            self.cursor.execute('''
            INSERT INTO claims_refs(claim_id, reference_id)
            VALUES($var,$var)'''.replace('$var','?'), (
                claim['id'],ref['hash']
            ))
        except sqlite3.IntegrityError as err:
            #db.rollback()
            pass
        finally:
            #self.db.commit()
            pass
    
    def extract_entity(self, e):
        for outgoing_property_id in e['claims'].values():
            for claim in outgoing_property_id:
                self.extract_claim(e['id'],claim)
                if 'references' in claim:
                    for ref in claim['references']: 
                        self.extract_claim_reference(claim, ref)
                        self.extract_reference(ref)

def claimParser(QID):
    entity_id = QID
    print('Setting up database ...')
    extractor = DatabaseExtractor()

    print('Fetching entity from API ...')
    entity = get_entity_dict_from_api(entity_id)

    if entity:
        print(f'Parsing entity: {entity_id}')
        extractor.extract_entity(entity)
    else:
        print(f'Failed to fetch entity: {entity_id}')

    extractor.finish_extraction()

def propertyFiltering(QID):
    reload(wdutils)
    DB_PATH = 'wikidata_claims_refs_parsed.db'
    claims_columns = ['entity_id','claim_id','rank','property_id','datatype','datavalue']

    properties_to_remove = {
        'general':[
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
        ],
        'specific': {}
    }

    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    # To see how many out of the total number of stored claims we are excluding by removing the general properties
    sql_query = "select count(*) from claims where property_id in $1;"
    sql_query = sql_query.replace('$1', '(' + ','.join([('"' + e + '"') for e in properties_to_remove['general']]) + ')')
    cursor.execute(sql_query)
    print('Removing the',len(properties_to_remove['general']),'properties deemed as ontological or unverbalisable')
    cursor = db.cursor()

    sql_query = "select * from claims where entity_id in $1;"
    sql_query = sql_query.replace('$1', '(' + ','.join([('"' + e + '"') for e in [QID]]) + ')')

    cursor.execute(sql_query)
    theme_df = pd.DataFrame(cursor.fetchall())
    theme_df.columns = claims_columns

    original_theme_df_size = theme_df.shape[0]
    last_stage_theme_df_size = original_theme_df_size

    print('-    Removing deprecated')

    # Remove deprecated
    theme_df = theme_df[theme_df['rank'] != 'deprecated'].reset_index(drop=True)
    print(
        '    -    Percentage of deprecated:',
        round((last_stage_theme_df_size-theme_df.shape[0])/original_theme_df_size*100, 2), '%'
    )
    last_stage_theme_df_size = theme_df.shape[0]

    print('-    Removing bad datatypes')

    # Remove external_ids, commonsMedia (e.g. photos), globe-coordinates, urls
    bad_datatypes = ['commonsMedia','external-id','globe-coordinate','url', 'wikibase-form',
                        'geo-shape', 'math', 'musical-notation', 'tabular-data', 'wikibase-sense']
    theme_df = theme_df[
        theme_df['datatype'].apply(
            lambda x : x not in bad_datatypes
        )
    ].reset_index(drop=True)
    print(
        '    -    Percentage of bad datatypes:',
        round((last_stage_theme_df_size-theme_df.shape[0])/original_theme_df_size*100, 2), '%'
    )
    last_stage_theme_df_size = theme_df.shape[0]

    print('-    Removing bad properties')

    # Remove specific properties such as P31 and P279
    theme_df = theme_df[
        theme_df['property_id'].apply(
            lambda x : (x not in properties_to_remove['general']))
        
    ].reset_index(drop=True)
    print(
        '    -    Percentage of ontology (non-domain) properties:',
        round((last_stage_theme_df_size-theme_df.shape[0])/original_theme_df_size*100, 2), '%'
    )
    last_stage_theme_df_size = theme_df.shape[0]

    print('-    Removing somevalue/novalue')

    # Remove novalue and somevalue
    theme_df = theme_df[
        theme_df['datavalue'].apply(
            lambda x : x not in ['somevalue', 'novalue']
        )
    ].reset_index(drop=True)
    print(
        '    -    Percentage of somevalue/novalue:',
        round((last_stage_theme_df_size-theme_df.shape[0])/original_theme_df_size*100, 2), '%'
    )
    last_stage_theme_df_size = theme_df.shape[0]

    print(
        'After all removals, we keep',
        round(last_stage_theme_df_size/original_theme_df_size*100, 2),
    )
    theme_df.to_sql('claims', db, if_exists='replace', index=False)

    return theme_df

def textualAugmentation(filtered_df):

    def get_object_label_given_datatype(row):
        def turn_to_century_or_millennium(y, mode):
            y = str(y)
            if mode == 'C':
                div = 100
                group = int(y.rjust(3, '0')[:-2])
                mode_name = 'century'
            elif mode == 'M':
                div = 1000
                group = int(y.rjust(4, '0')[:-3])
                mode_name = 'millenium'
            else:        
                raise ValueError('Use mode = C for century and M for millennium')
                
            if int(y)%div != 0:
                group += 1
            group = str(group)

            group_suffix = (
                'st' if group[-1] == '1' else (
                    'nd' if group[-1] == '2' else (
                        'rd' if group[-1] == '3' else 'th'
                    )
                )
            )

            return ' '.join([group+group_suffix, mode_name])

        dt = row['datatype']
        dv = row['datavalue']
        
        dt_types = ['wikibase-item', 'monolingualtext', 'quantity', 'time', 'string']
        if dt not in dt_types:
            print(dt)
            raise ValueError
        else:
            try:
                if dt == dt_types[0]:
                    return Wd_API.get_label(ast.literal_eval(dv)['value']['id'], True) #get label here
                elif dt == dt_types[1]:
                    dv = ast.literal_eval(dv)
                    return (dv['value']['text'], dv['value']['language'])
                elif dt == dt_types[2]:
                    dv = ast.literal_eval(dv)
                    amount, unit = dv['value']['amount'], dv['value']['unit']
                    if amount[0] == '+':
                        amount = amount[1:]
                    if str(unit) == '1':
                        return (str(amount), 'en')
                    else:
                        unit_entity_id = unit.split('/')[-1]
                        unit = Wd_API.get_label(unit_entity_id, True)#get label here
                        return (' '.join([amount, unit[0]]), unit[1])
                elif dt == dt_types[3]:
                    dv = ast.literal_eval(dv)
                    time = dv['value']['time']
                    timezone = dv['value']['timezone']
                    precision = dv['value']['precision']
                    assert dv['value']['after'] == 0 and dv['value']['before'] == 0

                    sufix = 'BC' if time[0] == '-' else ''
                    time = time[1:]

                    if precision == 11: #date
                        return (datetime.strptime(time, '%Y-%m-%dT00:00:%SZ').strftime('%d/%m/%Y') + sufix, 'en')
                    elif precision == 10: #month
                        try:
                            return (datetime.strptime(time, '%Y-%m-00T00:00:%SZ').strftime("%B of %Y") + sufix, 'en')
                        except ValueError:
                            return (datetime.strptime(time, '%Y-%m-%dT00:00:%SZ').strftime("%B of %Y") + sufix, 'en')
                    elif precision == 9: #year
                        try:
                            return (datetime.strptime(time, '%Y-00-00T00:00:%SZ').strftime('%Y') + sufix, 'en')
                        except ValueError:
                            return (datetime.strptime(time, '%Y-%m-%dT00:00:%SZ').strftime('%Y') + sufix, 'en')
                    elif precision == 8: #decade
                        try:
                            return (datetime.strptime(time, '%Y-00-00T00:00:%SZ').strftime('%Y')[:-1] +'0s' + sufix, 'en')
                        except ValueError:
                            return (datetime.strptime(time, '%Y-%m-%dT00:00:%SZ').strftime('%Y')[:-1] +'0s' + sufix, 'en')
                    elif precision == 7: #century
                        try:
                            parsed_time = datetime.strptime(time, '%Y-00-00T00:00:%SZ')
                        except ValueError:
                            parsed_time = datetime.strptime(time, '%Y-%m-%dT00:00:%SZ')
                        finally:                        
                            return (turn_to_century_or_millennium(
                                parsed_time.strftime('%Y'), mode='C'
                            ) + sufix, 'en')
                    elif precision == 6: #millennium
                        try:
                            parsed_time = datetime.strptime(time, '%Y-00-00T00:00:%SZ')
                        except ValueError:
                            parsed_time = datetime.strptime(time, '%Y-%m-%dT00:00:%SZ')
                        finally:                        
                            return (turn_to_century_or_millennium(
                                parsed_time.strftime('%Y'), mode='M'
                            ) + sufix, 'en')
                    elif precision == 4: #hundred thousand years 
                        timeint = int(datetime.strptime(time, '%Y-00-00T00:00:%SZ').strftime('%Y'))
                        timeint = round(timeint/1e5,1)
                        return (str(timeint) + 'hundred thousand years' + sufix, 'en')
                    elif precision == 3: #million years 
                        timeint = int(datetime.strptime(time, '%Y-00-00T00:00:%SZ').strftime('%Y'))
                        timeint = round(timeint/1e6,1)
                        return (str(timeint) + 'million years' + sufix, 'en')
                    elif precision == 0: #billion years 
                        timeint = int(datetime.strptime(time, '%Y-00-00T00:00:%SZ').strftime('%Y'))
                        timeint = round(timeint/1e9,1)
                        return (str(timeint) + 'billion years' +sufix, 'en')
                elif dt == dt_types[4]:
                    return (ast.literal_eval(dv)['value'], 'en')
            except ValueError as e:
                #pdb.set_trace()
                raise e
                
    def get_object_desc_given_datatype(row):
        dt = row['datatype']
        dv = row['datavalue']
        
        dt_types = ['wikibase-item', 'monolingualtext', 'quantity', 'time', 'string']
        if dt not in dt_types:
            print(dt)
            raise ValueError
        else:
            try:
                if dt == dt_types[0]:
                    return Wd_API.get_desc(ast.literal_eval(dv)['value']['id']) #get label here
                elif dt == dt_types[1]:
                    return ('no-desc', 'none')
                elif dt == dt_types[2]:
                    dv = ast.literal_eval(dv)
                    amount, unit = dv['value']['amount'], dv['value']['unit']
                    if amount[0] == '+':
                        amount = amount[1:]
                    if str(unit) == '1':
                        return ('no-desc', 'none')
                    else:
                        unit_entity_id = unit.split('/')[-1]
                        return Wd_API.get_desc(unit_entity_id)
                elif dt == dt_types[3]:
                    return ('no-desc', 'none')
                elif dt == dt_types[4]:
                    return ('no-desc', 'none')
            except ValueError as e:
                #pdb.set_trace()
                raise e
                
    def get_object_alias_given_datatype(row):
        dt = row['datatype']
        dv = row['datavalue']
        
        dt_types = ['wikibase-item', 'monolingualtext', 'quantity', 'time', 'string']
        if dt not in dt_types:
            print(dt)
            raise ValueError
        else:
            try:
                if dt == dt_types[0]:
                    return Wd_API.get_alias(ast.literal_eval(dv)['value']['id']) #get label here
                elif dt == dt_types[1]:
                    return ('no-alias', 'none')
                elif dt == dt_types[2]:
                    dv = ast.literal_eval(dv)
                    amount, unit = dv['value']['amount'], dv['value']['unit']
                    if amount[0] == '+':
                        amount = amount[1:]
                    if str(unit) == '1':
                        return ('no-alias', 'none')
                    else:
                        unit_entity_id = unit.split('/')[-1]
                        return Wd_API.get_alias(unit_entity_id)
                elif dt == dt_types[3]:
                    dv = ast.literal_eval(dv)
                    time = dv['value']['time']
                    timezone = dv['value']['timezone']
                    precision = dv['value']['precision']
                    assert dv['value']['after'] == 0 and dv['value']['before'] == 0

                    sufix = 'BC' if time[0] == '-' else ''
                    time = time[1:]

                    if precision == 11: #date
                        return ([
                            datetime.strptime(time, '%Y-%m-%dT00:00:%SZ').strftime('%-d of %B, %Y') + sufix,
                            datetime.strptime(time, '%Y-%m-%dT00:00:%SZ').strftime('%d/%m/%Y (dd/mm/yyyy)') + sufix,
                            datetime.strptime(time, '%Y-%m-%dT00:00:%SZ').strftime('%b %-d, %Y') + sufix
                        ], 'en')
                    else: #month
                        return ('no-alias', 'none')
                elif dt == dt_types[4]:
                    return ('no-alias', 'none')
            except ValueError as e:
                #pdb.set_trace()
                raise e

    Wd_API = wdutils.CachedWikidataAPI()
    Wd_API.languages = ['en']

    filtered_df['entity_label'] = filtered_df['entity_id'].apply(lambda x: Wd_API.get_label(x, True))
    filtered_df['entity_desc'] = filtered_df['entity_id'].apply(lambda x: Wd_API.get_desc(x))
    filtered_df['entity_alias'] = filtered_df['entity_id'].apply(lambda x: Wd_API.get_alias(x))

    print(' - Predicate augmentation...')
    filtered_df['property_label'] = filtered_df['property_id'].apply(lambda x: Wd_API.get_label(x, True))
    filtered_df['property_desc'] = filtered_df['property_id'].apply(lambda x: Wd_API.get_desc(x))
    filtered_df['property_alias'] = filtered_df['property_id'].apply(lambda x: Wd_API.get_alias(x))

    print(' - Object augmentation...')
    filtered_df['object_label'] = filtered_df.apply(get_object_label_given_datatype, axis=1)
    filtered_df['object_desc'] = filtered_df.apply(get_object_desc_given_datatype, axis=1)
    filtered_df['object_alias'] = filtered_df.apply(get_object_alias_given_datatype, axis=1)


    no_subject_label_perc = filtered_df[filtered_df['entity_label'].apply(lambda x: x[0] == 'no-label')].shape[0] / filtered_df.shape[0] * 100
    print(' - No subject label %:', no_subject_label_perc, '%')

    no_predicate_label_perc = filtered_df[filtered_df['property_label'].apply(lambda x: x[0] == 'no-label')].shape[0] / filtered_df.shape[0] * 100
    print(' - No predicate label %:', no_predicate_label_perc, '%')

    no_object_label_perc = filtered_df[filtered_df['object_label'].apply(lambda x: x[0] == 'no-label')].shape[0] / filtered_df.shape[0] * 100
    print(' - No object label %:', no_object_label_perc, '%')
    return filtered_df

def urlParser():
    Wd_API = wdutils.CachedWikidataAPI()
    Wd_API.languages = ['en']
    db = sqlite3.connect('wikidata_claims_refs_parsed.db')
    cursor = db.cursor()
    refs_columns = ['reference_id', 'reference_property_id', 'reference_index', 'reference_datatype', 'reference_value']
    cursor.execute('select * from refs where reference_datatype="url";')
    url_df = pd.DataFrame(cursor.fetchall())
    url_df.columns = refs_columns
    def reference_value_to_url(reference_value):
        if reference_value in ['novalue','somevalue']:
            return reference_value
        reference_value = ast.literal_eval(reference_value)
        assert reference_value['type'] == 'string'
        return reference_value['value']
    def reference_value_to_external_id(reference_value):
        if reference_value in ['novalue','somevalue']:
            return reference_value
        reference_value = ast.literal_eval(reference_value)
        assert reference_value['type'] == 'string'
        return reference_value['value']
    def get_formatter_url(entity_id):
        try:
            sparql_query = '''
                SELECT ?item ?itemLabel 
                WHERE 
                {
                wd:$1 wdt:P1630 ?item.
                SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
                }
            '''.replace('$1',entity_id)
            sparql_results = Wd_API.query_sparql_endpoint(sparql_query)
            if len(sparql_results['results']['bindings']) > 0:
                return sparql_results['results']['bindings'][0]['item']['value']
            else:
                return 'no_formatter_url'
        except Exception:
            print(entity_id)
            print(sparql_results)
            raise
    url_df['url'] = url_df.reference_value.apply(reference_value_to_url)
    cursor.execute('select * from refs where reference_datatype="external-id";')
    ext_id_df = pd.DataFrame(cursor.fetchall())
    ext_id_df.columns = refs_columns
    ext_id_df['ext_id'] = ext_id_df.reference_value.apply(reference_value_to_external_id)
    ext_id_df['formatter_url'] = ext_id_df['reference_property_id'].apply(get_formatter_url)
    ext_id_df['url'] = ext_id_df.apply(lambda x : x['formatter_url'].replace('$1', x['ext_id']), axis=1)
    columns_for_join = ['reference_id', 'reference_property_id','reference_index','reference_datatype','url']
    url_df_pre_join = url_df[columns_for_join]
    ext_id_df_pre_join = ext_id_df[columns_for_join]
    all_url_df = pd.concat([url_df_pre_join,ext_id_df_pre_join])
    all_url_df = all_url_df.sort_values(['reference_id','reference_index'])
    # drop those with url = 'no_formatter_url'
    all_url_df = all_url_df[all_url_df['url'] != 'no_formatter_url'].reset_index(drop=True)
    # drop those with url = somevalue and novalue
    all_url_df = all_url_df[~all_url_df['url'].isin(['somevalue','novalue'])]
    reference_id_counts = all_url_df.reference_id.value_counts().reset_index()
    reference_id_counts.columns = ['reference_id', 'counts']
    reference_id_counts_equal_1 = reference_id_counts[reference_id_counts['counts'] == 1].reference_id.tolist()
    all_url_df_eq1 = all_url_df[all_url_df.reference_id.isin(reference_id_counts_equal_1)]
    all_url_df_eq1 = all_url_df_eq1.reset_index(drop=True).drop('reference_index', axis=1)
    return all_url_df_eq1


if __name__ == '__main__':
    target_QID = 'Q42'
    #claimParser(target_QID) #save results in .db
    #filtered_df = propertyFiltering(target_QID) #update db and return dataframe after filtering
    #augmented_df = textualAugmentation(filtered_df) #textual information augmentation including label, desc, and alias
    #url_set = urlParser() #from ref table in .db
    