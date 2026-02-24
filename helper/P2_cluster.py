import gzip
import os
import pandas as pd
import numpy as np
import polars as pl
import argparse
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from joblib import Parallel, delayed

OUTPUT_NAME='CLDB_P2_hg19_022426.tsv'
INPUT_PATH='Z:/Members/clun/CLDB/meta/meta_SR_hg19.tsv'
REF_VER='hg19'
pd.set_option('display.expand_frame_repr', False)


def process_single_file(m_dict):
	"""Function to process one subject - runs in parallel"""
	P2_path = m_dict.get('P2_path')

	try:
		# Read file
		curr_df = pd.read_csv(P2_path, sep='\t', comment='#', header=None,
								usecols=[0, 1, 2, 7, 9], 
								names=['chrom1', 'pos1', 'SV_ID', 'info', 'genotype_raw'])
		
		# 1. Parsing chrom1
		curr_df['chrom1'] = curr_df['chrom1'].str.replace('chr', '') 
		
		# 2. Vectorized INFO parsing
		info_split = curr_df['info'].str.split(';', expand=True)
		# Note: Indexing by [3], [5], etc. is fast but risky if VCF order varies
		curr_df['SV_TYPE'] = info_split[3].str.replace('SVTYPE=', '')
		curr_df['SV_LEN'] = pd.to_numeric(info_split[2].str.replace('AVGLEN=', ''), errors='coerce')
		curr_df['chrom2'] = info_split[5].str.replace('CHR2=chr', '').str.replace('CHR2=', '')
		curr_df['pos2'] = pd.to_numeric(info_split[6].str.split('=').str[-1], errors='coerce')
		curr_df['genotype'] = curr_df['genotype_raw'].str.split(':').str[0]
		
		# 3. Add Metadata
		curr_df['PT_ID'] = m_dict.get('pt_id')
		curr_df['FAMILY'] = m_dict.get('family')
		curr_df['PROJECT'] = m_dict.get('project')
		curr_df['IS_PROBAND'] = m_dict.get('is_proband')
		curr_df['SYSTEM'] = m_dict.get('system')
		return curr_df
	except Exception as e:
		print(f"Error processing {m_dict.get('pt_id')}: {e}")
		return pd.DataFrame() # Return empty df on failure

def get_P2_df(meta_file, ref_ver):
	#global value for DBSCAN
	chrom_configs={
		'hg19': {	'1':249250621 ,	'2':243199373 ,	'3':198022430 ,	'4':191154276 ,
					'5':180915260,	'6':171115067 ,	'7':159138663 ,	'8':146364022 ,
					'9':141213431 ,	'10':135534747,	'11':135006516,	'12':133851895,
					'13':115169878,	'14':107349540,	'15':102531392,	'16':90354753 ,
					'17':81195210 ,	'18':78077248 ,	'20':63025520 ,	'19':59128983 ,
					'22':51304566 ,	'21':48129895,	'X':155270560,	'Y':59373566},
		'hg38':{	'1'	:248956422,	'2'	:242193529,	'3'	:198295559,	'4'	:190214555,
					'5'	:181538259,	'6'	:170805979,	'7'	:159345973,	'X'	:156040895,	
					'8'	:145138636,	'9'	:138394717,	'11':135086622,	'10':133797422,	
					'12':133275309,	'13':114364328,	'14':107043718,	'15':101991189,	
					'16':90338345,	'17':83257441,	'18':80373285,	'20':64444167,	
					'19':58617616,	'Y'	:57227415,	'22':50818468,	'21':46709983}
	}

	sizes = chrom_configs[ref_ver]
	size_df = pd.DataFrame(list(sizes.items()), columns=['Chromosome', 'Size'])
	size_df['offset'] = (size_df['Size'] + 10_000_000).cumsum().shift(fill_value=0)

	# print(size_df)

	# 2. Efficient Reading
	meta = pd.read_csv(meta_file, sep='\t')
	valid_meta = meta.dropna(subset=['P2_path']).to_dict('records')


	# n_jobs=-1 uses all available CPU cores
	# backend='multiprocessing' is usually best for CPU-heavy parsing
	results = Parallel(n_jobs=-1, backend='multiprocessing')(
		delayed(process_single_file)(m_dict) 
		for m_dict in tqdm(valid_meta, total=len(valid_meta), desc="Parallel Import P2 Data")
	)

    # Combine all individual dataframes into one
	df = pd.concat(results, ignore_index=True)

	# 3. Artifact Filtering (Vectorized)
	artifact_range = (32916144, 32916629) if ref_ver == 'hg38' else (33141211, 33141696)
	mask = (df['chrom1'] == '2') & (df['chrom2'] == '2') & \
			((df['pos1'].between(*artifact_range)) | (df['pos2'].between(*artifact_range)))
	df = df[~mask]

	df = df.merge(
		size_df[['Chromosome', 'offset']], 
		left_on='chrom1', 
		right_on='Chromosome', 
		how='left'
	).rename(columns={'offset': 'offset1'}).drop('Chromosome', axis=1)

	
	df = df.merge(
		size_df[['Chromosome', 'offset']], 
		left_on='chrom2', 
		right_on='Chromosome', 
		how='left'
	).rename(columns={'offset': 'offset2'}).drop('Chromosome', axis=1)

	
	df['cum_pos1'] = df['pos1'] + df['offset1']
	df['cum_pos2'] = df['pos2'] + df['offset2']


	#DBSCAN parameters
	epsilon = 500
	min_samples = 2
	cluster_counter = 0


	#DBSCAN by SV type
	grouped_SV_type = df.groupby('SV_TYPE')
	for name, group in grouped_SV_type:
		print(f'Clustering {name}')
		features = ['cum_pos1', 'cum_pos2']
		X = group[features]
		dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, n_jobs=-2)
		group['cluster_tmp'] = dbscan.fit_predict(X)
		group['cluster'] = np.where(group['cluster_tmp'] != -1, group['cluster_tmp']+cluster_counter+1, -1)
		cluster_counter = group['cluster'].max()
		df.loc[group.index, 'cluster'] = group['cluster']

	df=df.drop(['info', 'genotype_raw', 'cum_pos2', 'cum_pos1'], axis = 1)
	total_pt = len(df['PT_ID'].unique())

	#get count and propensity
	print('Getting cluster count')
	df['count'] = df.groupby(['cluster'])['cluster'].transform('count')
	df.loc[df['cluster'] == -1, 'count'] = 1
	df['cluster_propensity'] = df['count']/total_pt

	#get pseudo db freq
	print('Getting pseudo db freq')
	df['unique_pt_id_count'] = df.groupby('cluster')['PT_ID'].transform('nunique')
	df['unique_pt_id_count'] = np.where(df['cluster'] == -1, 1, df['unique_pt_id_count'])
	nsub=df.PT_ID.nunique()
	df['psuedo_df_freq'] = df['unique_pt_id_count']/nsub

	# 1. Calculate Totals
	total_proband_pt = df.loc[df['IS_PROBAND'] == True, 'PT_ID'].nunique()
	total_nonproband_pt = df.loc[df['IS_PROBAND'] == False, 'PT_ID'].nunique()

	# 2. Vectorized Proband Counts per Cluster
	# Filter out noise (CLUSTER_ID -1) and non-probands
	mask = (df['IS_PROBAND'] == 1) & (df['cluster'] != -1)
	proband_counts = df[mask].groupby('cluster')['PT_ID'].nunique()
	df['proband_only_count'] = df['cluster'].map(proband_counts)

	# Fill noise/singular cases: If -1, use IS_PROBAND as 1/0
	noise_mask = df['cluster'] == -1
	df.loc[noise_mask, 'proband_only_count'] = df['IS_PROBAND'].astype(int)
	df['proband_only_count'] = df['proband_only_count'].fillna(0).astype(int)
	df['proband_only_propensity'] = df['proband_only_count'] / total_proband_pt
	df['nonproband_only_count'] = df['unique_pt_id_count'] - df['proband_only_count']
	df['nonproband_only_propensity'] = df['nonproband_only_count'] / total_nonproband_pt
	return df



def main():
	P2 = get_P2_df(INPUT_PATH, REF_VER)
	print('Writing Results to TSV')
	OUTPUT_ORDER=['chrom1', 'pos1', 'chrom2', 'pos2', 'SV_ID', 'SV_TYPE', 'SV_LEN',
       'genotype', 'PT_ID', 'FAMILY', 'PROJECT', 'IS_PROBAND', 'SYSTEM',
       'cluster', 'count', 'cluster_propensity',
       'unique_pt_id_count', 'psuedo_df_freq', 'proband_only_count',
       'proband_only_propensity', 'nonproband_only_count',
       'nonproband_only_propensity']
	existing_cols = [c for c in OUTPUT_ORDER if c in P2.columns]
	P2_final = P2[existing_cols]

	rename_map = {
        'FAMILY': 'FAM_ID',
        'cluster': 'CLUSTER_ID',
        'count': 'COUNT',
        'cluster_propensity': 'CLUSTER_PROPENSITY',
        'unique_pt_id_count': 'UNIQUE_PT_COUNT',
        'psuedo_df_freq': 'PSEUDO_FREQ'
    }
	P2_final = P2_final.rename(columns=rename_map)
	print('converting to pl')
	pl_df = pl.from_pandas(P2_final)
	print('writing output')
	pl_df.write_csv(OUTPUT_NAME, separator='\t')
if __name__ == '__main__':
	main()