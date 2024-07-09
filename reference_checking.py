import sqlite3
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import yaml, json
from utils.verbalisation_module import VerbModule 
import nltk
from bs4 import BeautifulSoup
from cleantext import clean
from utils.sentence_retrieval_module import SentenceRetrievalModule
from utils.textual_entailment_module import TextualEntailmentModule
from tqdm import tqdm

class ReferenceChecker:
    def __init__(self, db_name: str = 'wikidata_claims_refs_parsed.db', config_path: str = 'config.yaml'):
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        self.config = self.load_config(config_path)
        self.verb_module = VerbModule()
        nltk.download('punkt', quiet=True)

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def execute_query(self, query: str, params: tuple = ()) -> List[tuple]:
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            return []

    def get_claim_df(self, entity_id: str) -> pd.DataFrame:
        query = """
        SELECT * FROM claim_text
        WHERE entity_id = ?
        """
        results = self.execute_query(query, (entity_id,))
        columns = [description[0] for description in self.cursor.description]
        return pd.DataFrame(results, columns=columns)
    
    def get_html_df(self, entity_id: str) -> pd.DataFrame:
        query = "SELECT * FROM html_text WHERE entity_id = ?"
        results = self.execute_query(query, (entity_id,))
        columns = [description[0] for description in self.cursor.description]
        return pd.DataFrame(results, columns=columns)

    def verbalisation(self, claim_df: pd.DataFrame) -> pd.DataFrame:
        triples = []
        for _, row in claim_df.iterrows():
            triple = {
                'subject': row['entity_label'],
                'predicate': row['property_label'],
                'object': row['object_label']
            }
            triples.append(triple)
        
        claim_df['verbalisation'] = self.verb_module.verbalise_triples(triples)
        claim_df['verbalisation_unks_replaced'] = claim_df['verbalisation'].apply(self.verb_module.replace_unks_on_sentence)
        claim_df['verbalisation_unks_replaced_then_dropped'] = claim_df['verbalisation'].apply(lambda x: self.verb_module.replace_unks_on_sentence(x, empty_after=True))
        
        return claim_df

    def sentenceSplitter(self, verbalised_claims_df_final, reference_text_df):
        join_df = pd.merge(verbalised_claims_df_final, reference_text_df[['reference_id', 'url', 'html']], on='reference_id', how='left')
        SS_df = join_df[['reference_id','url','verbalisation', 'html']].copy()
        def clean_html(html_content):
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)  
            cleaned_text = clean(text,
                                fix_unicode=True,  
                                to_ascii=True, 
                                lower=False,  
                                no_line_breaks=False,  
                                no_urls=True, 
                                no_emails=True,  
                                no_phone_numbers=True, 
                                no_numbers=False,  
                                no_digits=False, 
                                no_currency_symbols=True,  
                                no_punct=False, 
                                replace_with_url="",
                                replace_with_email="",
                                replace_with_phone_number="",
                                replace_with_number="",
                                replace_with_digit="",
                                replace_with_currency_symbol="")
            return cleaned_text
        def split_into_sentences(text):
            sentences = nltk.sent_tokenize(text)
            return sentences
        def slide_sentences(sentences, window_size=2):
            if len(sentences) < window_size:
                return [" ".join(sentences)]
            return [" ".join(sentences[i:i + window_size]) for i in range(len(sentences) - window_size + 1)]
        
        SS_df['html2text'] = SS_df['html'].apply(clean_html)
        SS_df['nlp_sentences'] = SS_df['html2text'].apply(split_into_sentences)
        SS_df['nlp_sentences_slide_2'] = SS_df['nlp_sentences'].apply(slide_sentences)

        return SS_df[['reference_id','verbalisation','url','nlp_sentences','nlp_sentences_slide_2']]
    
    def evidence_selection(self, splited_sentences_from_html: pd.DataFrame) -> pd.DataFrame:
        sr_module = SentenceRetrievalModule(max_len=self.config['evidence_selection']['token_size'])
        sentence_relevance_df = splited_sentences_from_html.copy()
        sentence_relevance_df.rename(columns={'verbalisation': 'final_verbalisation'}, inplace=True)

        def chunks(l: List, n: int) -> List[List]:
            n = max(1, n)
            return [l[i:i + n] for i in range(0, len(l), n)]
        
        def compute_scores(column_name: str) -> None:
            all_outputs = []
            for _, row in tqdm(sentence_relevance_df.iterrows(), total=sentence_relevance_df.shape[0]):
                outputs = []
                for batch in chunks(row[column_name], self.config['evidence_selection']['batch_size']):
                    batch_outputs = sr_module.score_sentence_pairs([(row['final_verbalisation'], sentence) for sentence in batch])
                    outputs += batch_outputs
                all_outputs.append(outputs)
            sentence_relevance_df[f'{column_name}_scores'] = pd.Series(all_outputs)
            assert all(sentence_relevance_df.apply(lambda x: len(x[column_name]) == len(x[f'{column_name}_scores']), axis=1))

        compute_scores('nlp_sentences')
        compute_scores('nlp_sentences_slide_2')

        def get_top_n_sentences(row: pd.Series, column_name: str, n: int) -> List[Dict]:
            sentences_with_scores = [{'sentence': t[0], 'score': t[1], 'sentence_id': f"{row.name}_{j}"} for j, t in enumerate(zip(row[column_name], row[f'{column_name}_scores']))]
            return sorted(sentences_with_scores, key=lambda x: x['score'], reverse=True)[:n]

        def filter_overlaps(sentences: List[Dict]) -> List[Dict]:
            filtered = []
            for evidence in sentences:
                if ';' in evidence['sentence_id']:
                    start_id, end_id = evidence['sentence_id'].split(';')
                    if not any(start_id in e['sentence_id'].split(';') or end_id in e['sentence_id'].split(';') for e in filtered):
                        filtered.append(evidence)
                else:
                    if not any(evidence['sentence_id'] in e['sentence_id'].split(';') for e in filtered):
                        filtered.append(evidence)
            return filtered
        
        def limit_sentence_length(sentence: str, max_length: int) -> str:
            return sentence[:max_length] + '...' if len(sentence) > max_length else sentence

        nlp_sentences_TOP_N, nlp_sentences_slide_2_TOP_N, nlp_sentences_all_TOP_N = [], [], []
        
        for _, row in tqdm(sentence_relevance_df.iterrows(), total=sentence_relevance_df.shape[0]):
            top_n = get_top_n_sentences(row, 'nlp_sentences', self.config['evidence_selection']['n_top_sentences'])
            top_n = [{'sentence': limit_sentence_length(s['sentence'], 1024), 'score': s['score'], 'sentence_id': s['sentence_id']} for s in top_n]
            nlp_sentences_TOP_N.append(top_n)
            
            top_n_slide_2 = get_top_n_sentences(row, 'nlp_sentences_slide_2', self.config['evidence_selection']['n_top_sentences'])
            top_n_slide_2 = [{'sentence': limit_sentence_length(s['sentence'], 1024), 'score': s['score'], 'sentence_id': s['sentence_id']} for s in top_n_slide_2]
            nlp_sentences_slide_2_TOP_N.append(top_n_slide_2)
            
            all_sentences = top_n + top_n_slide_2
            all_sentences_sorted = sorted(all_sentences, key=lambda x: x['score'], reverse=True)
            filtered_sentences = filter_overlaps(all_sentences_sorted)
            filtered_sentences = [{'sentence': limit_sentence_length(s['sentence'], 1024), 'score': s['score'], 'sentence_id': s['sentence_id']} for s in filtered_sentences]
            nlp_sentences_all_TOP_N.append(filtered_sentences[:self.config['evidence_selection']['n_top_sentences']])
        
        sentence_relevance_df['nlp_sentences_TOP_N'] = pd.Series(nlp_sentences_TOP_N)
        sentence_relevance_df['nlp_sentences_slide_2_TOP_N'] = pd.Series(nlp_sentences_slide_2_TOP_N)
        sentence_relevance_df['nlp_sentences_all_TOP_N'] = pd.Series(nlp_sentences_all_TOP_N)
        
        return sentence_relevance_df
    
    def textEntailment(self, evidence_df):
        SCORE_THRESHOLD=self.config['evidence_selection']['score_threshold']
        textual_entailment_df = evidence_df.copy()
        te_module = TextualEntailmentModule()

        keys = ['TOP_N', 'slide_2_TOP_N', 'all_TOP_N']
        te_columns = {f'evidence_TE_prob_{key}': [] for key in keys}
        te_columns.update({f'evidence_TE_prob_weighted_{key}': [] for key in keys})
        te_columns.update({f'evidence_TE_labels_{key}': [] for key in keys})
        te_columns.update({f'claim_TE_prob_weighted_sum_{key}': [] for key in keys})
        te_columns.update({f'claim_TE_label_weighted_sum_{key}': [] for key in keys})
        te_columns.update({f'claim_TE_label_malon_{key}': [] for key in keys})

        def process_row(row):
            claim = row['final_verbalisation']
            results = {}
            for key in keys:
                evidence = row[f'nlp_sentences_{key}']
                evidence_size = len(evidence)
                if evidence_size == 0:
                    results[key] = {
                        'evidence_TE_prob': [],
                        'evidence_TE_labels': [],
                        'evidence_TE_prob_weighted': [],
                        'claim_TE_prob_weighted_sum': [0, 0, 0],
                        'claim_TE_label_weighted_sum': 'NOT ENOUGH INFO',
                        'claim_TE_label_malon': 'NOT ENOUGH INFO'
                    }
                    continue

                evidence_TE_prob = te_module.get_batch_scores(
                    claims=[claim] * evidence_size,
                    evidence=[e['sentence'] for e in evidence]
                )

                evidence_TE_labels = [te_module.get_label_from_scores(s) for s in evidence_TE_prob]

                evidence_TE_prob_weighted = [
                    probs * ev['score'] for probs, ev in zip(evidence_TE_prob, evidence)
                    if ev['score'] > SCORE_THRESHOLD
                ]

                claim_TE_prob_weighted_sum = np.sum(evidence_TE_prob_weighted, axis=0) if evidence_TE_prob_weighted else [0, 0, 0]

                claim_TE_label_weighted_sum = te_module.get_label_from_scores(claim_TE_prob_weighted_sum) if evidence_TE_prob_weighted else 'NOT ENOUGH INFO'

                claim_TE_label_malon = te_module.get_label_malon(
                    [probs for probs, ev in zip(evidence_TE_prob, evidence) if ev['score'] > SCORE_THRESHOLD]
                )

                results[key] = {
                    'evidence_TE_prob': evidence_TE_prob,
                    'evidence_TE_labels': evidence_TE_labels,
                    'evidence_TE_prob_weighted': evidence_TE_prob_weighted,
                    'claim_TE_prob_weighted_sum': claim_TE_prob_weighted_sum,
                    'claim_TE_label_weighted_sum': claim_TE_label_weighted_sum,
                    'claim_TE_label_malon': claim_TE_label_malon
                }
            return results
        for i, row in tqdm(textual_entailment_df.iterrows(), total=textual_entailment_df.shape[0]):
            try:
                result_sets = process_row(row)
                for key in keys:
                    for k, v in result_sets[key].items():
                        te_columns[f'{k}_{key}'].append(v)
            except Exception as e:
                    logging.error(f"Error processing row {i}: {e}")
                    logging.error(f"Problematic row: {row}")
                    for key in keys:
                        for k in result_sets[key].keys():
                            te_columns[f'{k}_{key}'].append(None)


        for key in keys:
            for col in ['evidence_TE_prob', 'evidence_TE_prob_weighted', 'evidence_TE_labels',
                        'claim_TE_prob_weighted_sum', 'claim_TE_label_weighted_sum', 'claim_TE_label_malon']:
                textual_entailment_df[f'{col}_{key}'] = pd.Series(te_columns[f'{col}_{key}'])

        return textual_entailment_df

    def save_results_to_db(self, result: pd.DataFrame):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS checking_result (
            reference_id TEXT,
            final_verbalisation TEXT,
            url TEXT,
            nlp_sentences TEXT,
            nlp_sentences_slide_2 TEXT,
            nlp_sentences_scores TEXT,
            nlp_sentences_slide_2_scores TEXT,
            nlp_sentences_TOP_N TEXT,
            nlp_sentences_slide_2_TOP_N TEXT,
            nlp_sentences_all_TOP_N TEXT,
            evidence_TE_prob_TOP_N TEXT,
            evidence_TE_prob_weighted_TOP_N TEXT,
            evidence_TE_labels_TOP_N TEXT,
            claim_TE_prob_weighted_sum_TOP_N TEXT,
            claim_TE_label_weighted_sum_TOP_N TEXT,
            claim_TE_label_malon_TOP_N TEXT,
            evidence_TE_prob_slide_2_TOP_N TEXT,
            evidence_TE_prob_weighted_slide_2_TOP_N TEXT,
            evidence_TE_labels_slide_2_TOP_N TEXT,
            claim_TE_prob_weighted_sum_slide_2_TOP_N TEXT,
            claim_TE_label_weighted_sum_slide_2_TOP_N TEXT,
            claim_TE_label_malon_slide_2_TOP_N TEXT,
            evidence_TE_prob_all_TOP_N TEXT,
            evidence_TE_prob_weighted_all_TOP_N TEXT,
            evidence_TE_labels_all_TOP_N TEXT,
            claim_TE_prob_weighted_sum_all_TOP_N TEXT,
            claim_TE_label_weighted_sum_all_TOP_N TEXT,
            claim_TE_label_malon_all_TOP_N TEXT,
            PRIMARY KEY (reference_id, final_verbalisation, url)
        )
        """
        self.execute_query(create_table_query)

        upsert_query = """
        INSERT OR REPLACE INTO checking_result VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """

        for _, row in result.iterrows():
            values = [
                row['reference_id'],
                row['final_verbalisation'],
                row['url'],
                json.dumps(row['nlp_sentences']),
                json.dumps(row['nlp_sentences_slide_2']),
                json.dumps(row['nlp_sentences_scores'].tolist() if isinstance(row['nlp_sentences_scores'], np.ndarray) else row['nlp_sentences_scores']),
                json.dumps(row['nlp_sentences_slide_2_scores'].tolist() if isinstance(row['nlp_sentences_slide_2_scores'], np.ndarray) else row['nlp_sentences_slide_2_scores']),
                json.dumps(row['nlp_sentences_TOP_N']),
                json.dumps(row['nlp_sentences_slide_2_TOP_N']),
                json.dumps(row['nlp_sentences_all_TOP_N']),
                json.dumps([prob.tolist() if isinstance(prob, np.ndarray) else prob for prob in row['evidence_TE_prob_TOP_N']]),
                json.dumps([prob.tolist() if isinstance(prob, np.ndarray) else prob for prob in row['evidence_TE_prob_weighted_TOP_N']]),
                json.dumps(row['evidence_TE_labels_TOP_N']),
                json.dumps(row['claim_TE_prob_weighted_sum_TOP_N'].tolist() if isinstance(row['claim_TE_prob_weighted_sum_TOP_N'], np.ndarray) else row['claim_TE_prob_weighted_sum_TOP_N']),
                row['claim_TE_label_weighted_sum_TOP_N'],
                row['claim_TE_label_malon_TOP_N'],
                json.dumps([prob.tolist() if isinstance(prob, np.ndarray) else prob for prob in row['evidence_TE_prob_slide_2_TOP_N']]),
                json.dumps([prob.tolist() if isinstance(prob, np.ndarray) else prob for prob in row['evidence_TE_prob_weighted_slide_2_TOP_N']]),
                json.dumps(row['evidence_TE_labels_slide_2_TOP_N']),
                json.dumps(row['claim_TE_prob_weighted_sum_slide_2_TOP_N'].tolist() if isinstance(row['claim_TE_prob_weighted_sum_slide_2_TOP_N'], np.ndarray) else row['claim_TE_prob_weighted_sum_slide_2_TOP_N']),
                row['claim_TE_label_weighted_sum_slide_2_TOP_N'],
                row['claim_TE_label_malon_slide_2_TOP_N'],
                json.dumps([prob.tolist() if isinstance(prob, np.ndarray) else prob for prob in row['evidence_TE_prob_all_TOP_N']]),
                json.dumps([prob.tolist() if isinstance(prob, np.ndarray) else prob for prob in row['evidence_TE_prob_weighted_all_TOP_N']]),
                json.dumps(row['evidence_TE_labels_all_TOP_N']),
                json.dumps(row['claim_TE_prob_weighted_sum_all_TOP_N'].tolist() if isinstance(row['claim_TE_prob_weighted_sum_all_TOP_N'], np.ndarray) else row['claim_TE_prob_weighted_sum_all_TOP_N']),
                row['claim_TE_label_weighted_sum_all_TOP_N'],
                row['claim_TE_label_malon_all_TOP_N']
            ]
            self.execute_query(upsert_query, tuple(values))

        self.conn.commit()
        logging.info(f"Saved or updated {len(result)} rows in the checking_result table.")
        return 
    
    def TableMaking(self, verbalised_claims_df_final, result):
        verbalised_claims_df_final.set_index('reference_id', inplace=True)
        result.set_index('reference_id', inplace=True)
        results = pd.concat([verbalised_claims_df_final, result], axis=1)
        results['triple'] = results[['entity_label', 'property_label', 'object_label']].apply(lambda x: ', '.join(x), axis=1)
        all_result = pd.DataFrame()
        for idx, row in results.iterrows():
            aResult = pd.DataFrame(row["nlp_sentences_TOP_N"])[['sentence','score']]
            aResult.rename(columns={'score': 'Relevance_score'}, inplace=True)
            aResult = pd.concat([aResult, pd.DataFrame(row["evidence_TE_labels_all_TOP_N"], columns=['TextEntailment'])], axis=1)
            aResult = pd.concat([aResult, pd.DataFrame(np.max(row["evidence_TE_prob_all_TOP_N"], axis=1), columns=['Entailment_score'])], axis=1)
            aResult = aResult.reindex(columns=['sentence', 'TextEntailment', 'Entailment_score','Relevance_score'])
            aBox = pd.DataFrame({'triple': [row["triple"]], 'url': row['url'],'Results': [aResult]})
            all_result = pd.concat([all_result,aBox], axis=0)

        def dataframe_to_html(all_result):
            html = '<html><head><style>table {border-collapse: collapse; width: 100%;} th, td {border: 1px solid black; padding: 8px; text-align: left;} th {background-color: #f2f2f2;}</style></head><body>'
            for triple in all_result['triple'].unique():
                html += f'<h3>Triple: {triple}</h3>'
                df = all_result[all_result['triple']==triple].copy()
                for idx, row in df.iterrows():
                    url = row['url']
                    results = row['Results']
                    html += f'<h3>Reference: {url}</h3>'
                    html += results.to_html(index=False)
            html += '</body></html>'
            return html
        html_result = dataframe_to_html(all_result)
        return all_result, html_result

    
def main(qids: List[str]):
    with ReferenceChecker() as checker:
        result_tbs = pd.DataFrame()
        for qid in qids:
            claim_df = checker.get_claim_df(qid)
            html_df = checker.get_html_df(qid)
            verbalised_claims_df_final = checker.verbalisation(claim_df)
            splited_sentences_from_html = checker.sentenceSplitter(verbalised_claims_df_final, html_df)
            evidence_df = checker.evidence_selection(splited_sentences_from_html)
            result = checker.textEntailment(evidence_df)
            checker.save_results_to_db(result)
            all_result, html_result = checker.TableMaking(verbalised_claims_df_final, result)
            all_result['qid'] = qid
            result_tbs = pd.concat([result_tbs, all_result], axis=0)
        return result_tbs

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    qids = ["Q42"]
    result_tbs = main(qids)