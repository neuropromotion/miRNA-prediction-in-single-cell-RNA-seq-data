import pandas as pd
import numpy as np
import os

class Bootstrap:
    def __init__(self, mrna_path, mirna_path, barcodes_path, gene_lengths_path, 
                 selected_rna_path=None, selected_mirs=None, seed=None, log=True):
        self.log = log
        self.rng = np.random.default_rng(seed=seed)
        
        print("Loading data...")
        self.barcodes = pd.read_csv(barcodes_path, index_col=0)
        rna_raw = pd.read_csv(mrna_path, index_col=0)
        mir_raw = pd.read_csv(mirna_path, index_col=0)
        gene_lengths = pd.read_parquet(gene_lengths_path).set_index('gene_id')

        # Загрузка выбранных РНК из .txt (если путь указан)
        if selected_rna_path and os.path.isfile(selected_rna_path):
            with open(selected_rna_path, 'r') as f:
                self.selected_rna = [line.strip() for line in f if line.strip()]
        else:
            self.selected_rna = selected_rna_path # если передан список напрямую

        self.selected_mirs = selected_mirs

        # 1. Предварительная фильтрация РНК и длин (делаем один раз)
        shared_genes = sorted(list(set(rna_raw.index) & set(gene_lengths.index)))
        self.rna = rna_raw.loc[shared_genes]
        self.gene_lengths_kb = gene_lengths.loc[shared_genes, 'gene_length_kb']
        
        # 2. Предварительная агрегация миРНК (убираем 3p/5p один раз)
        print("Preprocessing miRNAs...")
        self.mir = self._preprocess_mirna(mir_raw)

        # 3. Синхронизация баркодов
        common_cells = sorted(list(set(self.rna.columns) & set(self.mir.columns) & set(self.barcodes['barcode'])))
        self.rna = self.rna[common_cells]
        self.mir = self.mir[common_cells]
        self.barcodes = self.barcodes[self.barcodes['barcode'].isin(common_cells)]
        
        self.tissues = sorted(self.barcodes['line'].unique().tolist())
        print(f"Ready. Cells: {len(common_cells)}, Tissues: {len(self.tissues)}")

    def _preprocess_mirna(self, df):
        df.index = df.index.str.lower()
        pattern = r'[-_](?:5p|3p)$'
        df.index = df.index.str.replace(pattern, '', regex=True)
        return df.groupby(level=0).sum()

    def get_tissues(self):
        return self.tissues

    def get_tissue_barcodes(self, tissue_type: str):
        if tissue_type not in self.get_tissues():
            raise ValueError('Unknown tissue type')
        tissue_specific_barcodes = self.barcodes[self.barcodes['line'].isin([tissue_type])]['barcode']
        return tissue_specific_barcodes

    def boot_fast(self, df, tissue_barcodes, K, N, tissue):
        """Быстрая векторная суммация."""
        data = df[tissue_barcodes].values
        # Генерируем матрицу индексов (K x N)
        idx = self.rng.choice(data.shape[1], size=(K, N), replace=True)
        
        boot_matrix = np.zeros((data.shape[0], N))
        for i in range(N):
            boot_matrix[:, i] = data[:, idx[:, i]].sum(axis=1)
            
        cols = [f'boot_K{K}_{tissue}_{i}' for i in range(N)]
        return pd.DataFrame(boot_matrix, index=df.index, columns=cols)

    def generate(self, tissue: str, K: int = 30, N: int = 50, subset_barcodes: list = None):
        if tissue not in self.tissues:
            raise ValueError(f"Unknown tissue: {tissue}")
            
        # 1. Получаем все баркоды этой ткани
        tissue_barcodes = self.barcodes[self.barcodes['line'] == tissue]['barcode'].tolist()
        
        # 2. Если передана подвыборка (train/test), фильтруем
        if subset_barcodes is not None:
            # Используем set для быстрого пересечения
            target_barcodes = list(set(tissue_barcodes) & set(subset_barcodes))
            if not target_barcodes:
                raise ValueError(f"No barcodes found for tissue '{tissue}' in the provided subset.")
        else:
            target_barcodes = tissue_barcodes

        n_cells = len(target_barcodes)

        # 3. Корректировка K
        if K > n_cells:
            print(f"WARNING: K={K} > n_cells({n_cells}) in subset. Setting K={n_cells}")
            K = n_cells

        
            
        # 4. Бутстреп (используем нашу быструю функцию)
        rna_boot = self.boot_fast(self.rna, target_barcodes, K, N, tissue)
        mir_boot = self.boot_fast(self.mir, target_barcodes, K, N, tissue)

        # 6. Фильтрация по генам/миРНК до нормализации
        if self.selected_rna:
            rna_boot = rna_boot.loc[rna_boot.index.intersection(self.selected_rna)]
        if self.selected_mirs:
            mir_boot = mir_boot.loc[mir_boot.index.intersection(self.selected_mirs)]
            
        current_lengths = self.gene_lengths_kb.loc[rna_boot.index]
        # 5. Нормализация (TPM для РНК, CPM для миРНК)
        rpk = rna_boot.div(current_lengths, axis=0)
        tpm = rpk.div(rpk.sum(axis=0), axis=1) * 1e6
        
        cpm = mir_boot.div(mir_boot.sum(axis=0), axis=1) * 1e6
        
        if self.log:
            tpm = np.log2(tpm + 1)
            cpm = np.log2(cpm + 1)
        
            
        return tpm, cpm

