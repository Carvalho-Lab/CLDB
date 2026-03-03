import pandas as pd
import numpy as np
import sqlite3 as sq
from sqlalchemy import create_engine
import os
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend

REF_VER = 'hg38'
INPUT_META=f'Z:/Members/clun/CLDB/meta/meta_SR_{REF_VER}.tsv'
DB_PATH = 'prod_CLDB_SR.sqlite'
OUTPUT_DIR = f'./data/4.0.0/CLDB_CGR_{REF_VER}'



def query(pt_id, conn):
    # Use the passed connection instead of creating a new one every time
    cnv = pd.read_sql(f'SELECT UUID, chrom1 AS chr, pos1 AS start, chrom2, pos2 AS end, SV_ID AS CNV_ID, SV_TYPE AS CNV_TYPE, SV_LEN AS CNV_LEN, PT_ID, FAM_ID, IS_PROBAND, PROJECT, CLUSTER_ID, COUNT, UNIQUE_PT_COUNT, PSEUDO_FREQ, SD_overlap, OMIM_count, RefSeq_count from CNV_{REF_VER} where 1=1 AND pt_id = "{pt_id}"', conn)
    p2 = pd.read_sql(f'SELECT chrom1 AS chr, pos1 AS start, pos2 AS end, SV_ID, SV_TYPE from P2_{REF_VER} where 1=1 AND pt_id = "{pt_id}" AND SV_TYPE != "BND"', conn)
    p2['chr']=p2['chr'].astype(str)
    cnv['chr']=cnv['chr'].astype(str)
    print(len(cnv))
    print(len(p2))
    return p2, cnv

def process_cnv(p2, cnv):
    # Filter p2
    p2 = p2[abs(p2['start'] - p2['end']) > 1000]
    p2 = p2.reset_index(drop=True)
    # print(p2)
    p2_dict={}
    for name, group in p2.groupby('chr'):
        p2_dict[name] = group
    # Process start positions
# 2. Map CNV rows to P2 indices
    final_idx_left = []
    final_idx_right = []

    for idx, row in cnv.iterrows():
        _chr = row['chr']
        
        # If chromosome is missing, append None to keep the lists aligned with cnv index
        if _chr not in p2_dict:
            final_idx_left.append(None)
            final_idx_right.append(None)
            continue
            
        # Closest for Start
        abs_diff_s_s = abs(row['start'] - p2_dict[_chr]['start'])
        abs_diff_s_e = abs(row['start'] - p2_dict[_chr]['end'])
        final_idx_left.append(abs_diff_s_s.idxmin() if abs_diff_s_s.min() < abs_diff_s_e.min() else abs_diff_s_e.idxmin())
        
        # Closest for End
        abs_diff_e_s = abs(row['end'] - p2_dict[_chr]['start'])
        abs_diff_e_e = abs(row['end'] - p2_dict[_chr]['end'])
        final_idx_right.append(abs_diff_e_s.idxmin() if abs_diff_e_s.min() < abs_diff_e_e.min() else abs_diff_e_e.idxmin())

    # 3. Build the DataFrames using the gathered indices
    # We use reindex() because it handles None/missing values by filling with NaN automatically
    left_df = p2.reindex(final_idx_left).reset_index(drop=True)
    left_df.columns = 'left_' + left_df.columns
    
    right_df = p2.reindex(final_idx_right).reset_index(drop=True)
    right_df.columns = 'right_' + right_df.columns

    # 4. Use JOIN
    # Ensure cnv has a clean index to join against
    cnv = cnv.reset_index(drop=True)
    merged_df = cnv.join([left_df, right_df])
    # 5. Calculations (Vectorized is better than apply here)
    merged_df['left_width'] = abs(merged_df['left_start'] - merged_df['left_end'])
    merged_df['right_width'] = abs(merged_df['right_start'] - merged_df['right_end'])
    # Calculate distances
    merged_df['left_dist'] = merged_df.apply(lambda x: min(abs(x['start']-x['left_start']), abs(x['start']-x['left_end'])), axis=1)
    merged_df['right_dist'] = merged_df.apply(lambda x: min(abs(x['start']-x['right_start']), abs(x['start']-x['right_end'])), axis=1)

    # Matching criteria
    merged_df['match_SV'] = (merged_df['left_SV_ID'] == merged_df['right_SV_ID']).astype(int)
    merged_df['match_left_SV_size'] = np.where(abs(merged_df['CNV_LEN'] - merged_df['left_width']) < 2000, 1,0)
    merged_df['match_right_SV_size'] = (abs(merged_df['CNV_LEN'] - merged_df['right_width']) < 2000).astype(int)
    merged_df['match_left_SV_TYPE'] = (((merged_df['CNV_TYPE'] == 'DUP') | (merged_df['CNV_TYPE'] == 'TRP')) & (merged_df['left_SV_TYPE'] == 'DUP')) | (((merged_df['CNV_TYPE'] == 'HET_DEL') | (merged_df['CNV_TYPE'] == 'HOM_DEL')) & (merged_df['left_SV_TYPE'] == 'DEL')).astype(int)
    merged_df['match_right_SV_TYPE'] = (((merged_df['CNV_TYPE'] == 'DUP') | (merged_df['CNV_TYPE'] == 'TRP')) & (merged_df['right_SV_TYPE'] == 'DUP')) | (((merged_df['CNV_TYPE'] == 'HET_DEL') | (merged_df['CNV_TYPE'] == 'HOM_DEL')) & (merged_df['right_SV_TYPE'] == 'DEL')).astype(int)
    merged_df['match_CNV_SV'] = ((merged_df['match_SV'] == 1) & (merged_df['match_left_SV_TYPE'] == 1) & (merged_df['match_left_SV_size'] == 1)).astype(int)
    merged_df = merged_df.sort_values(by='chr')
    merged_df.rename(columns={'chr':'chrom1', 'start':'pos1', 'end':'pos2', 'CNV_ID': 'SV_ID', 'CNV_TYPE': 'SV_TYPE', 'CNV_LEN': 'SV_LEN'}, inplace=True)
    return merged_df


def worker_task(pt_id, db_path, output_dir):
    output_file = f'{output_dir}/{pt_id}.tsv'
    if os.path.exists(output_file):
        return None

    conn = sq.connect(db_path)
    p2, cnv = query(pt_id, conn)
                
    if p2.empty or cnv.empty:
        return None
    out = process_cnv(p2, cnv)
    if out.empty:
        return None
    conn.close()
    out.to_csv(f'{output_dir}/{pt_id}.tsv', sep='\t', index=False)
    return pt_id

def main():
    # Load metadata
    print(f"INPUT METADATA: {INPUT_META}")
    print(f"DATABASE: {DB_PATH}")
    print(f"OUTPUT DIRECTORY: {OUTPUT_DIR}")

    df = pd.read_csv(INPUT_META, sep='\t')
    df = df[df["CNV_outlier"] == 0]
    df = df[['pt_id', 'P2_path']]
    df=df.dropna().reset_index()
    pt_ids=sorted(df['pt_id'])
    print(len(pt_ids))
    
    for pt_id in pt_ids:
        output_file = f'{OUTPUT_DIR}/{pt_id}.tsv'
        if os.path.exists(output_file):
            pass
        else:
            print(pt_id)

    results = Parallel(n_jobs=-1)(
    delayed(worker_task)(pt_id, DB_PATH, OUTPUT_DIR) 
    for pt_id in tqdm(pt_ids, desc="Processing Patients")
)
                    
if __name__ == '__main__':
    main()