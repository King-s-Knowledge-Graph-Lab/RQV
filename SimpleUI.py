import gradio as gr
import Wikidata_Text_Parser as wtr
import sqlite3
import Prove_lite as prv
import pandas as pd

def wtr_process(qid):
    try:
        conn = sqlite3.connect('wikidata_claims_refs_parsed.db')
        target_QID = qid

        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='claims'")
        table_exists = cursor.fetchone()

        if table_exists:
            cursor.execute("SELECT entity_id FROM claims WHERE entity_id=?", (target_QID,))
            result = cursor.fetchone()
            
            if result:
                print(f"{target_QID} already exists in the 'claims' table. Skipping execution.")
            else:
                progress = gr.Progress(0)
                progress(0.00, desc="Wikidata claims parsing...")
                wtr.claimParser(target_QID) #save results in .db
                filtered_df = wtr.propertyFiltering(target_QID) #update db and return dataframe after filtering
                progress(0.25, desc="URL and HTML parsing...")
                url_set = wtr.urlParser() #from ref table in .db
                html_set = wtr.htmlParser(url_set) #Original html docs collection
                progress(0.50, desc="claim2Text...")
                claim_text = wtr.claim2text(html_set) #Claims generation
                progress(0.74, desc="html2Text...")
                html_text = wtr.html2text(html_set)
                claim_text = claim_text.astype(str)
                html_text = html_text.astype(str)
                claim_text.to_sql('claim_text', conn, if_exists='replace', index=False)
                html_text.to_sql('html_text', conn, if_exists='replace', index=False)
                progress(1, desc="completed...")
        else:
            progress = gr.Progress(0)
            progress(0.00, desc="Wikidata claims parsing...")
            wtr.claimParser(target_QID) #save results in .db
            filtered_df = wtr.propertyFiltering(target_QID) #update db and return dataframe after filtering
            progress(0.25, desc="URL and HTML parsing...")
            url_set = wtr.urlParser() #from ref table in .db
            html_set = wtr.htmlParser(url_set) #Original html docs collection
            progress(0.50, desc="claim2Text...")
            claim_text = wtr.claim2text(html_set) #Claims generation
            progress(0.74, desc="html2Text...")
            html_text = wtr.html2text(html_set)
            claim_text = claim_text.astype(str)
            html_text = html_text.astype(str)
            claim_text.to_sql('claim_text', conn, if_exists='replace', index=False)
            html_text.to_sql('html_text', conn, if_exists='replace', index=False)
            progress(1, desc="completed...")


        query = f"""
            SELECT
                claim_text.entity_label,
                claim_text.property_label,
                claim_text.object_label,
                html_text.url
            FROM claim_text
            INNER JOIN html_text ON claim_text.reference_id = html_text.reference_id
            WHERE claim_text.entity_id = '{target_QID}'
        """

        result_df = pd.read_sql_query(query, conn)

        conn.commit()
        conn.close()

        return result_df
    except Exception as e:
            error_df = pd.DataFrame({'Error': [str(e)]})
            return error_df


def prv_process(qid):
    conn = sqlite3.connect('wikidata_claims_refs_parsed.db')

    query = f"""
    SELECT html_text.*
    FROM html_text
    INNER JOIN claim_text ON html_text.reference_id = claim_text.reference_id
    WHERE claim_text.entity_id = '{qid}'
"""
    reference_text_df = pd.read_sql_query(query, conn)
    query = f"SELECT * FROM claim_text WHERE entity_id = '{qid}'"
    claim_df = pd.read_sql_query(query, conn)
    
    progress = gr.Progress(0)
    BATCH_SIZE = 256
    N_TOP_SENTENCES = 5
    SCORE_THRESHOLD = 0
    TEST_SIZE = 5 #Sampling for Easy Test
    progress(0.33, desc="Verbalising claims...")
    verbalised_claims_df_final = prv.verbalisation(claim_df)
    verbalised_claims_df_final = verbalised_claims_df_final.sample(n=TEST_SIZE, replace=False) #data sampling for easy-test
    progress(0.66, desc="Relevance sentence selection...")
    sentence_relevance_df = prv.RelevantSentenceSelection(verbalised_claims_df_final, reference_text_df, BATCH_SIZE, N_TOP_SENTENCES)
    progress(0.99, desc="Text entailment...")
    result = prv.textEntailment(sentence_relevance_df, SCORE_THRESHOLD)
    result.to_csv('results.csv', index=False, encoding='utf-8-sig')
    result.to_sql('result_table', conn, if_exists='replace', index=False)
    conn.close()
    return result



with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Reference Quality Verification Tool
        This is a tool for verifying the reference quality of Wikidata claims related to the target entity item.
        Parsing could take 3~5 mins depending on the number of references.
        """
    )
    inp = gr.Textbox(label="Input QID", placeholder="Input QID (i.e. Q42)")
    out = gr.Dataframe(label="Parsing result (not presenting parsed HTMLs)",  headers=["entity_label", "property_label", "object_label", "url"])
    run_button_1 = gr.Button("Start parsing")
    run_button_1.click(wtr_process, inp, out)


    gr.Markdown(
        """
        Text entailment !
        """
    )
    out_2 = gr.DataFrame(label="Text entailment result")

    run_button_2 = gr.Button("Start processing")
    run_button_2.click(prv_process, inp, out_2)

    run_button_3 = gr.Button("View previous result")
    run_button_3.click(prv_process, inp, out_2)

    
if __name__ == "__main__":
    demo.launch()