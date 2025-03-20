import pandas as pd
import numpy as np
import scipy
import re
import os
from io import StringIO
import requests
import dash
from dash import dcc, html, Input, Output, State

np.random.seed(42)

# Load data
PUB_DATA_URL = "https://raw.githubusercontent.com/zanemit/QS-papers-citations-interactive/main/data/publication_data.csv"
response_pub = requests.get(PUB_DATA_URL)
if response_pub.status_code==200:
    pub_df = pd.read_csv(StringIO(response_pub.text), index_col=0)
else:
    print("Failed to download publication data!")

ASJC_DATA_URL = "https://raw.githubusercontent.com/zanemit/QS-papers-citations-interactive/main/data/asjc_data.csv"
response_asjc = requests.get(ASJC_DATA_URL)
if response_pub.status_code==200:
    asjc_df = pd.read_csv(StringIO(response_asjc.text), index_col=0)
else:
    print("Failed to download asjc data!")

# Dash app setup
app = dash.Dash(__name__)
server = app.server

# DEFAULT PARAMETERS
DEFAULT_ITERS = 100
DEFAULT_SELF_CITATION_FRACTION = 0.37

app.layout = html.Div([
    html.H1("Publikāciju un citējumu skaita simulācija QS reitingiem"),

    # assumptions
    html.H3("Pieņēmumi"),
    html.Label("Cik zemākās kvartiles (Q4) publikācijām ekvivalenta ir viena Q3 publikācija? Ievadiet skaitli starp 1 un 20: "),
    dcc.Input(type="number", value=2, id="q4-q3", step=0.5, min=1, max=20),
    html.Br(),

    html.Label("Cik Q4 publikācijām ekvivalenta ir viena Q2 publikācija? Ievadiet skaitli starp 1 un 20: "),
    dcc.Input(type="number", value=4, id="q4-q2", step=0.5, min=1, max=20),
    html.Br(),

    html.Label("Cik Q4 publikācijām ekvivalenta ir viena augstākās kvartiles (Q1) publikācija? Ievadiet skaitli starp 1 un 20: "),
    dcc.Input(type="number", value=8, id="q4-q1", step=0.5, min=1, max=20),
    html.Br(),

    # params
    html.H3("Hiperparametri"),
    html.Label("Par cik procentiem samazināt Q4 publikāciju skaitu?"),
    dcc.Slider(0, 100, 5, value=100, id='q4-slider'),

    html.Label("Par cik procentiem samazināt Q3 publikāciju skaitu?"),
    dcc.Slider(0, 100, 5, value=50, id='q3-slider'),
    
    html.Label("Par cik procentiem samazināt Q2 publikāciju skaitu?"),
    dcc.Slider(0, 100, 5, value=20, id='q2-slider'),

    html.Label("Pašcitējumu daļa. Ievadiet decimālskaitli starp 0 un 1: "),
    dcc.Input(type="number", value=0.37, id="self-cite", step=0.01, min=0, max=1),
    html.Br(),

    html.Label("Simulācijas iterācijas (ar 100 iterācijām jāuzgaida līdz pusminūtei): "),
    dcc.Input(type="number", value=DEFAULT_ITERS, id="iters-input"),

    html.Br(), html.Br(),
    html.Button("Simulēt rezultātus", id="run-button", n_clicks=0),

    html.H3("Rezultāti"),
    html.Pre(id="output-results")  # Output display
])

def generate_truncated_normal(size, mean, lower, upper, std_dev=1):
          # Compute the truncated normal distribution parameters
          a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
          return scipy.stats.truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=size)

def merge_publications_with_asjc(publication_data, asjc_data):
    publication_df = publication_data.copy()
    asjc_df = asjc_data.copy()

    # ###### SEPARATE PAPERS IN NICHE JOURNALS (one ASJC code) ######
    # Preliminary merge -- on journal title
    publication_df = publication_df.merge(
        asjc_df.loc[:,['Source Title', 'All Science Journal Classification Codes (ASJC)']],
        on='Source Title', how='left')

    # Fallback merge
    missing_asjc_mask = publication_df['All Science Journal Classification Codes (ASJC)'].isna()
    print(f"{missing_asjc_mask.sum()} papers could not be matched based on journal title!")
    fallback_merge = publication_df.loc[missing_asjc_mask].drop(columns=['All Science Journal Classification Codes (ASJC)']).merge(
                    asjc_df.loc[:, ['ISSN', 'All Science Journal Classification Codes (ASJC)']],
                    on='ISSN',  how='left')
    publication_df.loc[missing_asjc_mask, 'All Science Journal Classification Codes (ASJC)'] = fallback_merge.loc[:,'All Science Journal Classification Codes (ASJC)'].values

    # Fallback merge 2: try removing brackets from source titles
    missing_asjc_mask = publication_df['All Science Journal Classification Codes (ASJC)'].isna()
    print(f"{missing_asjc_mask.sum()} papers could not be matched even after considering ISSN!")
    publication_df.loc[missing_asjc_mask, 'Source Title'] = publication_df.loc[missing_asjc_mask,'Source Title'].apply(lambda text: re.sub(r'\(.*?\)', '', text).strip())   # keep only stuff before brackets
    fallback_merge2 = publication_df.loc[missing_asjc_mask].drop(columns=['All Science Journal Classification Codes (ASJC)']).merge(
                    asjc_df.loc[:, ['Source Title', 'All Science Journal Classification Codes (ASJC)']],
                    on='Source Title',  how='left')
    publication_df.loc[missing_asjc_mask, 'All Science Journal Classification Codes (ASJC)'] = fallback_merge2.loc[:,'All Science Journal Classification Codes (ASJC)'].values
    print(f"{publication_df['All Science Journal Classification Codes (ASJC)'].isna().sum()} papers could not be matched even after removing brackets!")

    # Not all titles could be matched with a title in asjc_df
    publication_df = publication_df.dropna(subset='All Science Journal Classification Codes (ASJC)')

    return publication_df

def compute_citations_per_paper(publication_data_asjc, self_citation_fraction):
    """
    publication_data_asjc (pd dataframe) : publication data with asjc codes merged
                      (output of `merge_publications_with_asjc`)
    self_citation_fraction (float) : number between 0 and 1
    """
    if self_citation_fraction <0 or self_citation_fraction >1:
        raise ValueError("Self-citation fraction must be a number between 0 and 1")
    pub_df = publication_data_asjc.copy()

    # ###### COMPUTE CITATION NUMBERS PER PAPER ######
    pub_df['Journal ASJC count'] = pub_df['All Science Journal Classification Codes (ASJC)'].apply(lambda s: len(s)/8)

    pub_df_niche = pub_df.loc[pub_df['Journal ASJC count']==1, :].reset_index()
    pub_df_regular = pub_df.loc[pub_df['Journal ASJC count']>1, :].reset_index()
    # print(f"There are {pub_df_niche.shape[0]} publications in niche journals and {pub_df_regular.shape[0]} publications elsewhere.")

    CPP_raw = 0.667*pub_df_niche['Citations'].sum()/pub_df_niche.shape[0] +\
                          0.333*pub_df_regular['Citations'].sum()/pub_df_regular.shape[0]

    CPP = CPP_raw*(1-self_citation_fraction)

    return CPP

def compute_H(publication_data_asjc, self_citation_fraction):
    # publication_df, asjc_df, self_citation_fraction):
    """
    self-citation fraction should be a float (not a list/arr)
    """
    if self_citation_fraction <0 or self_citation_fraction >1:
        raise ValueError("Self-citation fraction must be a number between 0 and 1")
    pub_df = publication_data_asjc

    pub_df['Journal ASJC count'] = pub_df['All Science Journal Classification Codes (ASJC)'].apply(lambda s: len(s))
    pub_df_niche = pub_df.loc[pub_df['Journal ASJC count']==1, :].reset_index()

    valid_citation_fraction = 1-self_citation_fraction
    h0 = 1  ;  h1 = 1
    while ((pub_df['Citations']*valid_citation_fraction)>h0).sum() > h0:
        h0+=1

    while ((pub_df_niche['Citations']*valid_citation_fraction)>h1).sum() > h1:
        h1+=1

    H = (h0*0.33) + (h1*0.67)

    return H

def simulate_publications_and_citations(publication_data, asjc_data, 
                                        PERC_TO_LOWER_Q4, PERC_TO_LOWER_Q3, 
                                        PERC_TO_LOWER_Q2, Q4_TO_Q3_EQUIVALENT=2, 
                                        Q4_TO_Q2_EQUIVALENT=4, Q4_TO_Q1_EQUIVALENT=8,
                                 iters=1000, SELF_CITATION_FRACTION=0.37):
    pub_df = merge_publications_with_asjc(publication_data, asjc_data)

    pub_df = pub_df.loc[:, [
        'journal_quartile', 'Citations', 'All Science Journal Classification Codes (ASJC)'
        ]].copy()

    # hyperparameter dictionaries
    equivalent_paper_dict={
        '4_3': Q4_TO_Q3_EQUIVALENT,
        '4_2': Q4_TO_Q2_EQUIVALENT,
        '4_1': Q4_TO_Q1_EQUIVALENT,
        '3_2': Q4_TO_Q2_EQUIVALENT/Q4_TO_Q3_EQUIVALENT,
        '3_1': Q4_TO_Q1_EQUIVALENT/Q4_TO_Q3_EQUIVALENT,
        '2_1': Q4_TO_Q1_EQUIVALENT/Q4_TO_Q2_EQUIVALENT,
    }

    perc_to_replace_dict={
        4: PERC_TO_LOWER_Q4,
        3: PERC_TO_LOWER_Q3,
        2: PERC_TO_LOWER_Q2
    }

    possible_quartiles = np.arange(1,5)

    final_papers = np.empty(iters) * np.nan
    final_citations = np.empty(iters) * np.nan
    simulated_H = np.empty(iters) * np.nan
    simulated_CPP = np.empty(iters) * np.nan

    for it in range(iters):
        # CURRENT PAPER NUMS
        papers_per_quartile = pub_df['journal_quartile'].value_counts().to_frame()

        if it%10==0:
            print(f"Running iteration {it}...")

        if it==0:
            initial_papers = papers_per_quartile.loc[:, 'count'].sum()
            initial_citations = pub_df['Citations'].sum()
            initial_CPP = compute_citations_per_paper(pub_df, SELF_CITATION_FRACTION)
            initial_H = compute_H(pub_df, SELF_CITATION_FRACTION)
            print(f"Current papers: {initial_papers}\n")
            print(f"Current citations: {initial_citations}\n")
            print(f"Current Citations per Paper: {initial_CPP}\n")
            print(f"Current H-Index: {initial_H}\n")

        for quartile_to_replace in [4,3,2]:
            # COMPUTE THE NUMBER OF PAPERS THAT SHOULD BE REDISTRIBUTED
            papers_per_quartile.loc[f'Q{quartile_to_replace}', 'target_count'] = int(papers_per_quartile.loc[f'Q{quartile_to_replace}', 'count']*(1 - (perc_to_replace_dict[quartile_to_replace]*0.01)))
            papers_per_quartile.loc[f'Q{quartile_to_replace}', 'redistribute_count'] = papers_per_quartile.loc[f'Q{quartile_to_replace}', 'count'] - papers_per_quartile.loc[f'Q{quartile_to_replace}', 'target_count']

            # GENERATE PAPER QUARTILES THAT THE CURRENT QUARTILE WILL BE REPLACED WITH
            replacement_options = np.setdiff1d(possible_quartiles, [quartile_to_replace])
            replacement_options = replacement_options[replacement_options<quartile_to_replace]  # can only replace with a better quartile
            replacement_quartile_arr = np.random.choice(replacement_options, size=10000)
            replacement_paper_arr = np.array([equivalent_paper_dict[f'{quartile_to_replace}_{x}'] for x in replacement_quartile_arr])

            replacement_quartiles = replacement_quartile_arr[np.cumsum(replacement_paper_arr)<=papers_per_quartile.loc[f'Q{quartile_to_replace}', 'redistribute_count']]
            replacement_papers = replacement_paper_arr[np.cumsum(replacement_paper_arr)<=papers_per_quartile.loc[f'Q{quartile_to_replace}', 'redistribute_count']]

            # with the random quartile choice, some papers might not get replaced - DEAL WITH THOSE
            num_nonreplaced_papers = papers_per_quartile.loc[f'Q{quartile_to_replace}', 'redistribute_count']-replacement_papers.sum()
            replacement_options_papers = np.array([equivalent_paper_dict[f'{quartile_to_replace}_{x}'] for x in replacement_options])

            papers_to_add_options = replacement_options_papers[replacement_options_papers<=num_nonreplaced_papers]
            if len(papers_to_add_options)>0:
                papers_to_add = papers_to_add_options.max()
                quartile_to_add_papers_to = replacement_options[np.argwhere(replacement_options_papers==papers_to_add)][0][0]
                replacement_papers = np.concatenate((replacement_papers, [papers_to_add]))
                replacement_quartiles = np.concatenate((replacement_quartiles, [quartile_to_add_papers_to]))

            # add the papers to the count
            for quart in replacement_quartiles:
                papers_per_quartile.loc[f"Q{quart}", 'count'] += 1
            papers_per_quartile.loc[f"Q{quartile_to_replace}", 'remaining_count'] = papers_per_quartile.loc[f"Q{quartile_to_replace}", 'count'] - replacement_papers.sum()

        papers_per_quartile.loc['Q1', 'remaining_count'] = papers_per_quartile.loc['Q1', 'count']

        final_papers[it] = papers_per_quartile.loc[:, 'remaining_count'].sum()

        for i, quart in enumerate(papers_per_quartile.index):
            pub_df_sub = pub_df.loc[pub_df['journal_quartile']==quart, :]
            sampled_indices = np.random.choice(pub_df_sub.index, size=int(papers_per_quartile.loc[quart, 'remaining_count']))
            sampled_df = pub_df_sub.loc[sampled_indices].copy()

            papers_per_quartile.loc[quart, 'citations'] = sampled_df['Citations'].sum()

            if i==0:
                final_sampled_df = sampled_df
            else:
                final_sampled_df = pd.concat((final_sampled_df, sampled_df), ignore_index=True)

        final_citations[it] = papers_per_quartile.loc[:, 'citations'].sum()

        # GENERATE RANKING METRICS
        simulated_CPP[it] = compute_citations_per_paper(final_sampled_df, SELF_CITATION_FRACTION)
        simulated_H[it] = compute_H(final_sampled_df, SELF_CITATION_FRACTION)

    return initial_papers, initial_citations, initial_CPP, initial_H, final_papers, final_citations, simulated_CPP, simulated_H

@app.callback(
    Output("output-results", "children"),
    Input("run-button", "n_clicks"),
    [
        Input("q4-slider", "value"), 
        Input("q3-slider", "value"), 
        Input("q2-slider", "value"), 
        Input("q4-q3", "value"), 
        Input("q4-q2", "value"), 
        Input("q4-q1", "value"), 
        Input("self-cite", "value"), 
        Input("iters-input", "value")
     ]
)

def run_simulation(n_clicks, q4_slider, q3_slider, q2_slider, q4_q3, q4_q2, q4_q1, self_cite, iters_input):
    if n_clicks==0:
            return "Nospiediet `Simulēt rezultātus!`"
    
    initial_papers, initial_citations, initial_CPP, initial_H, final_papers, final_citations, simulated_CPP, simulated_H = simulate_publications_and_citations(
         publication_data=pub_df,
         asjc_data=asjc_df,
         PERC_TO_LOWER_Q4=q4_slider,
         PERC_TO_LOWER_Q3=q3_slider,
         PERC_TO_LOWER_Q2=q2_slider,
         Q4_TO_Q3_EQUIVALENT=q4_q3,
         Q4_TO_Q2_EQUIVALENT=q4_q2,
         Q4_TO_Q1_EQUIVALENT=q4_q1,
         SELF_CITATION_FRACTION=self_cite,
         iters=iters_input
    ) 

    return html.Div([
         html.P(f"Paredzamais publikāciju skaits: {final_papers.mean():.0f} ± {(final_papers.std()/np.sqrt(len(final_papers))):.0f}. Procentuālās izmaiņas: {100*((final_papers.mean()/initial_papers)-1):.1f}%."),
         html.P(f"Paredzamais citējumu skaits: {final_citations.mean():.0f} ± {(final_citations.std()/np.sqrt(len(final_citations))):.0f}. Procentuālās izmaiņas: {100*((final_citations.mean()/initial_citations)-1):.1f}%."),
         html.P(f"Paredzamais 'Citations per Paper' rādītājs: {simulated_CPP.mean():.1f} ± {(simulated_CPP.std()/np.sqrt(len(simulated_CPP))):.1f}. Procentuālās izmaiņas: {100*((simulated_CPP.mean()/initial_CPP)-1):.1f}%."),
         html.P(f"Paredzamais 'H-index': {simulated_H.mean():.1f} ± {(simulated_H.std()/np.sqrt(len(simulated_H))):.1f}. Procentuālās izmaiņas: {100*((simulated_H.mean()/initial_H)-1):.1f}%."),
        html.P("\nATJAUNINIET LAPU, LAI SIMULĒTU VĒLREIZ!")
    ])

if __name__ == '__main__':
     port = int(os.environ.get('PORT', 8050))
     app.run(debug=True, host='0.0.0.0', port=port)
