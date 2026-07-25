import pandas as pd
import numpy as np
from collections import Counter

def TPM(df_input, length_file_path):
    df = df_input.copy()
    gene_length = pd.read_parquet(length_file_path)
    gene_length.index = gene_length.pop('gene_id')

    features = df.index.tolist()
    gene_length=gene_length.loc[features]
    gene_length.insert(0, 'gene_id', gene_length.index)
    lengths = gene_length.set_index('gene_id')['gene_length_kb']
    L = lengths.reindex(df.index) # double check
    
    rpk_gtex = df.div(L, axis=0)
    scale = rpk_gtex.sum(axis=0).replace(0, np.nan)
    tpm = rpk_gtex.div(scale, axis=1) * 1e6
    tpm = tpm.fillna(0.0)
    return tpm

def CPM(df_input):
    df = df_input.copy()
    counts_per_sample = df.sum(axis=0)
    cpm = df.divide(counts_per_sample, axis=1) * 1e6
    cpm = cpm.fillna(0.0)
    return cpm

def return_sc_mirs(sc_file_path):
    df_mir = pd.read_csv(sc_file_path, index_col=0)
    df_mir.index = df_mir.index.str.replace('_', '-', regex=False).str.lower()

    return df_mir.index.tolist()

def shared_mrnas(gtex_file_path, tcga_file_path, sc_file_path, gene_length_file_path, save=False):
    del_suff = lambda x: x.split('.')[0]
    # GTEx genes
    df_genes_gtex = pd.read_csv(
        gtex_file_path,
        sep='\t',
        skiprows=2,
        low_memory=False
    )
    df_genes_gtex['Name'] = df_genes_gtex['Name'].apply(del_suff)
    gtex_rnas = df_genes_gtex['Name'].tolist()

    # TCGA genes
    df_rna_tcga = pd.read_csv(tcga_file_path)
    df_rna_tcga['gene_id'] = df_rna_tcga['gene_id'].apply(del_suff)
    tcga_rnas = df_rna_tcga['gene_id'].tolist()

    # scRNA-seq genes
    rna_raw = pd.read_csv(sc_file_path, index_col=0) 
    sc_rnas = rna_raw.index.tolist()

    # genes with known length 
    gene_length = pd.read_parquet(gene_length_file_path)
    gene_length.index = gene_length.pop('gene_id')
    gene_length_mirs = gene_length.index.tolist()

    # shared genes
    shared_rnas = set(gtex_rnas) & set(tcga_rnas) & set(sc_rnas) & set(gene_length_mirs)
    shared_rnas = sorted(list(shared_rnas))

    if save:
        with open('features.txt', 'w') as f:
            for rna in shared_rnas:
                f.write(f'{rna}\n')
    return shared_rnas

def prepare_GTEx_mir(gtex_file_path, URS_file_path, annotation_file_path, shared_mirs=None, return_mirs=False):
    # load data and rename URS labels to miRNA names
    df_mir = pd.read_table(gtex_file_path)
    URS = pd.read_csv(URS_file_path, index_col=0) # URS miRNA + name
    df_mir = df_mir.rename(columns={'Unnamed: 0' : 'miRNA'})
    df_mir.index = df_mir.pop('miRNA')
    df_mir = df_mir[df_mir.index.str.contains('URS', na=False)]
    URS['miRNA_name'] = URS['miRNA_name'].str.lower()
    df_mir = df_mir.loc[URS['URS']]

    rename_URS = {}
    for urs in df_mir.index.tolist():
        rename_URS.setdefault(urs, None)
        rename_URS[urs] = (URS[URS['URS'] == urs]['miRNA_name'].item()).lower()

    df_mir = df_mir.rename(index=rename_URS)
    df_mir.index = df_mir.index.str.replace('_', '-', regex=False).str.lower()
    
    if return_mirs:
        return df_mir.index.tolist()

   # filter for shared miRNAs before calculating CPM
    if shared_mirs is not None:
        df_mir = df_mir.loc[shared_mirs]

    # calculate CPM
    cpm = CPM(df_mir)

    # load annotation and merge with CPM
    set_mir = cpm.columns.tolist()
    cols = ['SAMPID', 'SMTS', 'SMTSD']

    annotation = pd.read_table(annotation_file_path, low_memory=False)
    mir_annot_cols = annotation[annotation['SAMPID'].isin(set_mir)][cols]

    cpm = cpm.transpose() 
    cpm.insert(0, 'SAMPID', cpm.index.tolist()) 

    cpm = pd.merge(
        cpm,
        mir_annot_cols[['SAMPID', 'SMTSD']],
        on='SAMPID',
        how='left',      
        validate='one_to_one'  
    )

    cpm.insert(0, 'SMTSD', cpm.pop('SMTSD'))
    cpm = cpm.sort_values('SMTSD')  
    cpm['SMTSD'] = (
        cpm.groupby('SMTSD')
        .cumcount()
        .astype(str)
        .radd(cpm['SMTSD'] + '_')
    )

    # remove duplicate columns
    temp = dict(Counter(cpm.columns.tolist()))
    temp = [k for k, v in temp.items() if v > 1]
    cpm = cpm.loc[:, ~cpm.columns.isin(temp)]

    cpm.index = cpm.pop('SMTSD')
    cpm.drop(columns=['SAMPID'], inplace=True)
    cpm = cpm.transpose()
    return cpm

def prepare_TCGA_mir(tcga_file_path, shared_mirs=None, return_mirs=False):
    df_mir = pd.read_csv(tcga_file_path)
    df_mir.index = df_mir.pop('miRNA_ID')
    df_mir.index = df_mir.index.str.replace('_', '-', regex=False).str.lower()
    
    if return_mirs:
        return df_mir.index.tolist()

    if shared_mirs is not None:
        df_mir = df_mir.loc[shared_mirs]

    cpm = CPM(df_mir)
    return cpm

def prepare_GTEx_rna(gtex_file_path, annotation_file_path, gene_length_file_path, features=None):
    del_suff = lambda x: x.split('.')[0]

    df_genes_gtex = pd.read_csv(
        gtex_file_path,
        sep='\t',
        skiprows=2,
        low_memory=False
    )

    df_genes_gtex['Name'] = df_genes_gtex['Name'].apply(del_suff)
    df_genes_gtex.index = df_genes_gtex.pop('Name')

    if features is not None:
        df_genes_gtex = df_genes_gtex.loc[features]
    df_genes_gtex.pop('Description')

    df_genes_gtex = df_genes_gtex.T
    df_genes_gtex.insert(0, 'SAMPID', df_genes_gtex.index)

    annotation = pd.read_table(annotation_file_path, low_memory=False)

    df_genes_gtex = pd.merge(
        df_genes_gtex,
        annotation[annotation['SAMPID'].isin(df_genes_gtex.index)][['SAMPID', 'SMTSD']],
        on='SAMPID',
        how='left',      
        validate='one_to_one'  
    )
    df_genes_gtex.insert(1, 'SMTSD', df_genes_gtex.pop('SMTSD'))
    df_genes_gtex = df_genes_gtex.sort_values(by='SMTSD')

    df_genes_gtex['SMTSD'] = (
        df_genes_gtex.groupby('SMTSD')
        .cumcount()
        .astype(str)
        .radd(df_genes_gtex['SMTSD'] + '_')
    )
    df_genes_gtex.pop('SAMPID')

    df_genes_gtex.index = df_genes_gtex.pop('SMTSD')
    df_genes_gtex = df_genes_gtex.transpose()

    tpm = TPM(df_genes_gtex, gene_length_file_path)

    return tpm

def prepare_TCGA_rna(tcga_file_path, gene_length_file_path, features=None):
    del_suff = lambda x: x.split('.')[0]

    df_rna_tcga = pd.read_csv(tcga_file_path)

    df_rna_tcga['gene_id'] = df_rna_tcga['gene_id'].apply(del_suff)
    df_rna_tcga.index = df_rna_tcga.pop('gene_id')

    if features is not None:
        df_rna_tcga = df_rna_tcga.loc[features]

    tpm = TPM(df_rna_tcga, gene_length_file_path)

    return tpm