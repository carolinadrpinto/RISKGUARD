
## Vou guardar as coisas como a Carolina Vasconcelos, em ficheiros arrow, só com as colunas que me interessam e com  as incoerências tratadas

import pandas as pd
import polars as pl
from datetime import datetime
from copy import deepcopy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List


SEED=42

import random, torch
random.seed(SEED)
torch.manual_seed(SEED)
from transformers import AutoTokenizer, AutoModel



from collections import Counter


import unicodedata

def clean_text(text, solos=False, pontuation = False, lower= True, accents = False, stopwords = None, remove_numbers=False):
    
    if lower:
        text =  text.lower()
        
    if stopwords:
        tokens = text.split()
        tokens_filtrados = [t for t in tokens if t not in stopwords]
        text = " ".join(tokens_filtrados)
    

    # remover números, traços, pontos de interrogação, tags de html, remover \n e \, /, remover pontuação quando antecede ou precede um número
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('\n', ' ').replace('\t', ' ')

    # remover barras
    #text = re.sub(r'[-–—/_\\|]', ' ', text)

    # remover pontos de interrogação que aparecem random e aspas deste tipo - ''
    text = text.replace('?', '')
    text = text.replace('¿', '')
    text = text.replace("'", "")

    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    if pontuation:
        text = re.sub(r'[^\w\s]', '', text) # remover pontuação
    
    if accents:
        nfkd = unicodedata.normalize('NFD', text)
        text = ''.join([c for c in nfkd if not unicodedata.combining(c)])
    
    if solos:
        text = ' '.join(word for word in text.split() if len(word) > 2)

    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"\s*\.\s*\.\s*$", ".", text)  # final com ". ."
    
    return text


def prepro_obj(df_contratos : pl.DataFrame, stopwords : list):
    "preprocessar a coluna do objeto e do tipo de contrato"
    "Guarda também as tokens do objeto e do tipo de contrato"

    return df_contratos.with_columns(
        # coloca tudo em lower, remove caracteres especiais e tags de html. Coloca missings como strings vazias
        # boa base para fazer por cima outro tipo de preprcessamento
        pl.col("Objeto").map_elements(lambda x: 
                               clean_text(x, lower = True, remove_numbers=False, pontuation=False) 
                               if x is not None else "",
                                return_dtype=pl.Utf8) \
                                .alias("Objeto_LIMPO"),
        # coloca tudo em lower, remove stopwords, números, pontuação e caracteres estranhos
        pl.col("Objeto").map_elements(lambda x: 
                               clean_text(x, lower = True, stopwords=stopwords, remove_numbers=True, solos=True, pontuation=True) 
                               if x is not None else "",
                                return_dtype=pl.Utf8) \
                               .alias("Objeto_LIMPO_2"),
        pl.col("Tipo(s) de contrato").map_elements(lambda x: x.lower() if pd.notna(x) else '', return_dtype=pl.Utf8) \
            .alias("Tipo(s) de contrato_LIMPO")
    )


# EMBEDDINGS

def generate_embeddings_serafim335(
    df: pl.DataFrame,
    text_column: str,
    batch_size: int = 32,
    device: str = None,
) -> torch.Tensor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    texts = df.select(text_column).to_series().to_list()
    tokenizer = AutoTokenizer.from_pretrained('PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir')
    model = AutoModel.from_pretrained('PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir').to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts,
                               return_tensors="pt",
                               padding=True,
                               truncation=True,
                               max_length=512).to(device)
            outputs = model(**inputs)
            # Pooling (mean over tokens)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize (as in LaBSE paper)
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.append(batch_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)



def generate_embeddings_labse(
    df: pl.DataFrame,
    text_column: str,
    batch_size: int = 32,
    device: str = None,
) -> torch.Tensor:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    texts = df.select(text_column).to_series().to_list()
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
    model = AutoModel.from_pretrained("sentence-transformers/LaBSE").to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts,
                               return_tensors="pt",
                               padding=True,
                               truncation=True,
                               max_length=512).to(device)
            outputs = model(**inputs)
            # Pooling (mean over tokens)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize (as in LaBSE paper)
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.append(batch_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)





def cosine_similarity(dfA, dfB, eps=1e-8):
    if dfA.shape != dfB.shape:
        raise ValueError(f"Shapes diferentes: {dfA.shape} vs {dfB.shape}")
    
    A = dfA.to_numpy()
    B = dfB.to_numpy()


    num  = (A * B).sum(axis=1)
    den  = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + eps
    cos  = np.full(A.shape[0], np.nan, dtype="float32")
    cos = num / den

    cos_dis = 1.0 - cos

    return cos_dis


def make_min_ratio_strategy(y, min_ratio=0.1):
    counts = Counter(y)
    majority = max(counts.values())
    min_n = int(majority * min_ratio)
    return {cls: max(n, min_n) for cls, n in counts.items() if n < min_n}




def distribution(metric_col_name, df, tipo_filter=None):

    if tipo_filter is not None:
        df=df[df['Tipo(s) de contrato']==tipo_filter]
    
    metric_col = pd.to_numeric(df[metric_col_name], errors='coerce')

    plt.figure(figsize=(7,4))
    plt.hist(metric_col, density=True, alpha=0.8)
    #plt.axvline(metric_col.mean(), linestyle='--', linewidth=1.5, label=f"média = {metric_col.mean():.3f}")
    plt.axvline(metric_col.mean()+3*metric_col.std(), linestyle=':', color='#004F7F',linewidth=1.5)
    plt.axvline(metric_col.mean()-3*metric_col.std(), linestyle=':', color='#0000CD',linewidth=1.5)
    if tipo_filter is not None:
        plt.title(f"Cossine Distance {metric_col_name} - Tipo: {tipo_filter}")
    else:
        plt.title(f"Cossine Distance {metric_col_name}")
    plt.xlabel("cos_dis (1 - cos_sim)")
    plt.xlim(0, 2)   # distância do cosseno vai de 0 a 2
    plt.legend()
    plt.tight_layout()
    plt.show()



    
def flag(df, metric_col):
    df_copy = deepcopy(df)
    lower_limit = df_copy[metric_col].mean()-3*df_copy[metric_col].std()
    upper_limit = df_copy[metric_col].mean()+3*df_copy[metric_col].std()
    conds = [df_copy[metric_col]<lower_limit, 
             df_copy[metric_col].between(lower_limit, upper_limit, inclusive="both"),
             df_copy[metric_col]>upper_limit]
    choices = [1, 0, 1]
    df_copy[f"flag_{metric_col}"] = np.select(conds, choices, default=0).astype('int8')
    return df_copy


def flag_simples(row):
    if (row['Objeto_LIMPO']==row['Tipo(s) de contrato'].lower() or row['Objeto_LIMPO']=='aquisição de bens'):
        return 'igual ao tipo'
    else:
        return 'ok'
    

def flag_mal_definidos(df, row, metric_col):
    lower_limit = df[metric_col].mean()-3*df[metric_col].std()
    upper_limit = df[metric_col].mean()+3*df[metric_col].std()

    if row[metric_col]<lower_limit:
        return 'muito parecido ao tipo'

    elif row[metric_col]>upper_limit:
        return 'muito parecido ao tipo'

    return 'ok'



def flag_mal_definidos(row, metric_col, lower_limit, upper_limit):
    if row[metric_col]<lower_limit:
        return 1

    # elif row[metric_col]>upper_limit:
    #     return 1
    else:
        return 0
    


def jaccard_similarity(s1, s2):
    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())
    return len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0

def dice_similarity(s1, s2):
    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())
    return 2 * len(set1 & set2) / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0


import string

def preprocess_text(text: str) -> str:
    if text is None:
        return text

    # 1. Remover pontuação ASCII
    #text = text.translate(str.maketrans("", "", string.punctuation))

    text = text.translate(str.maketrans(
    string.punctuation,          # caracteres a substituir
    " " * len(string.punctuation) # todos substituídos por espaço
    ))

    # 3. Normalizar espaços
    text = re.sub(r"\s+", " ", text).strip()

    # # 4. Remover palavras com 1 ou 2 caracteres
    # text = ' '.join(word for word in text.split() if len(word) > 2)

    # 5. Normalizar novamente espaços (após remoção)
    text = re.sub(r"\s+", " ", text).strip()

    return text




def extract_sliding_windows(years: np.ndarray, window_shape: int) -> List[np.ndarray]:
    """Extracts sliding windows from an array of years

    In case where the number of years is less than the window size, the function returns a truncated array.
    
    Example:
        >>> extract_sliding_windows(np.array([2023, 2022, 2021, 2020]), 3)
        >>> [array([2023, 2022, 2021]), array([2022, 2021, 2020]), array([2021, 2020]), array([2020])]
    
    Args:
        years (np.ndarray): An array of years, descending order (i.e. higher years have lower indexes)
        window_shape (int): The window size to be considered

    Returns:
        np.ndarray: An array of arrays, where the sub-arrays are sliding windows over the years
    """

    windows = []

    for i in range(0, np.size(years)):
        if i+window_shape < np.size(years):
            windows.append(years[i:i+window_shape])
        else:
            windows.append(years[i:])
            
    return windows


def reverse_order_unique_contract_years(contracts_table: pl.DataFrame, contract_year_column_name: str) -> np.ndarray:
    """Extracts unique years in descending order from a pl.DataFrame column

    Args:
        contracts_table (pl.DataFrame): The contracts table holding a column with contract years
        contract_year_column_name (str): The name of the column holding contract years

    Returns:
        np.ndarray: An array of contract years in descending order
    """
    unique_years = contracts_table \
        .with_columns(pl.col("Data Celebração").dt.year().alias(contract_year_column_name)) \
        .filter(pl.col(contract_year_column_name).is_not_null()) \
        .unique(subset=[contract_year_column_name]) \
        .select(contract_year_column_name) \
        .to_numpy() \
        .flatten()
    
    return np.sort(unique_years)[::-1]


def mean_per_year_per_cpv(contracts: pl.DataFrame, window: np.array, contract_year_column: str = "contract_year") -> pl.DataFrame:

    # para cada window quero filtrar pelo intervalo e calcular a média
    return contracts.filter((pl.col(contract_year_column)>=window[-1]) & (pl.col(contract_year_column)<=window[0])) \
                .group_by(["cpvs"]) \
                .agg((pl.col("Preço Contratual (€)").mean().round(3)).alias("Preço Estimado")) \
                .with_columns(
                    pl.lit(window[0]).alias("end_period").cast(pl.Int64)
                )


def get_covid_legal_frameworks() -> List[str]:
    return ["10-A/2020, de 13.03",
        "1-A/2020, de 20.03",
        "30/2021, de 21.05"]



def preprocess_text(text: str) -> str:
    if text is None:
        return text

    # 1. Remover pontuação ASCII
    #text = text.translate(str.maketrans("", "", string.punctuation))

    text = text.translate(str.maketrans(
    string.punctuation,          # caracteres a substituir
    " " * len(string.punctuation) # todos substituídos por espaço
    ))

    # 3. Normalizar espaços
    text = re.sub(r"\s+", " ", text).strip()

    # # 4. Remover palavras com 1 ou 2 caracteres
    # text = ' '.join(word for word in text.split() if len(word) > 2)

    # 5. Normalizar novamente espaços (após remoção)
    text = re.sub(r"\s+", " ", text).strip()

    return text