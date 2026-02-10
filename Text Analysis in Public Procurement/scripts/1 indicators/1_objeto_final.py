import os

import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns


from copy import deepcopy

import spacy

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# Set random seed for reproducibility
SEED=42
np.random.seed(SEED)

from utilities.funcoes import cosine_similarity, make_min_ratio_strategy, distribution, preprocess_text
from utilities.schema import contracts_schema, lots_schema


import re
import string

import time
from pathlib import Path

from collections import Counter


contratos = pl.read_parquet(
    "../../data/contratos_cleaned_prepro.parquet",
    columns=[
        "N.º Procedimento (ID BASE)", "N.º Contrato", "Tipo(s) de contrato_LIMPO", "Objeto_LIMPO", "Objeto_LIMPO_2", "Objeto", "Tipo de procedimento"
    ]
    )


# preprocessar o objeto para medir o tamanho e para passar no pos tagging
# ver a distribuição do tamanho do objeto por ano, por tipo de procedimento, 
# por entidade adjudicatária, por local de Local de execução das principais prestações objeto do contrato
contratos = contratos.with_columns(
    pl.col("Objeto_LIMPO").map_elements(preprocess_text, return_dtype=pl.Utf8).alias("Objeto_LIMPO_Final")
)

objeto_prepro = contratos.select(["Objeto_LIMPO_Final", "Objeto","N.º Procedimento (ID BASE)", "N.º Contrato"]).to_pandas()
objeto_prepro["Objeto_len"] = objeto_prepro["Objeto_LIMPO_Final"].str.split().str.len()
contratos = contratos.with_columns(
    pl.Series("objeto_len", objeto_prepro["Objeto_len"].to_numpy())
)




## READ EMBEDDINGS

embeddings_serafim_objeto = pl.read_parquet("../../data/embeddings/serafim/Objeto_LIMPO.parquet")
embeddings_serafim_tipo = pl.read_parquet("../../data/embeddings/serafim/Tipo(s) de contrato_LIMPO.parquet")

embeddings_labse_objeto = pl.read_parquet("../../data/embeddings/labse/Objeto_LIMPO.parquet")
embeddings_labse_tipo = pl.read_parquet("../../data/embeddings/labse/Tipo(s) de contrato_LIMPO.parquet")


# Sinalizar contratos sem letras e com um tamanho muito grando - significa que este campo tem clausulas contratuais
contratos = contratos.with_columns(
    pl.when(pl.col("Objeto").str.contains("[A-Za-z]")).then(pl.lit(0)).otherwise(pl.lit(1)).alias("flag_1011_anom")
).with_columns(
    pl.when((pl.col("objeto_len")>300) & (pl.col("flag_1011_anom")==0)).then(pl.lit(1)).otherwise(pl.lit(0)).alias("flag_1011_anom")
)

# não queria que a distância fosse condicionada por estas "anomalias", ou seja, que o desvio padrão fosse condicionado por estas anomalias

# calcular a distância do objeto ao tipo de contrato
cos_serafim = cosine_similarity(embeddings_serafim_tipo.drop(["N.º Procedimento (ID BASE)", "N.º Contrato"]), embeddings_serafim_objeto.drop(["N.º Procedimento (ID BASE)", "N.º Contrato"]))
cos_labse = cosine_similarity(embeddings_labse_tipo.drop(["N.º Procedimento (ID BASE)", "N.º Contrato"]), embeddings_labse_objeto.drop(["N.º Procedimento (ID BASE)", "N.º Contrato"]))

contratos = contratos.with_columns(
    pl.Series("dist_serafim_obj_tipo", cos_serafim),
    pl.Series("dist_labse_obj_tipo", cos_labse)
)

# sinalizar os que estão fora do thereshold

k = 3

stats = contratos.select([
    pl.col("dist_serafim_obj_tipo").mean().alias("mu_serafim"),
    pl.col("dist_serafim_obj_tipo").std().alias("sigma_serafim"),
    pl.col("dist_labse_obj_tipo").mean().alias("mu_labse"),
    pl.col("dist_labse_obj_tipo").std().alias("sigma_labse"),
]).row(0)

mu_serafim, sigma_serafim, mu_labse, sigma_labse = stats
upper_serafim, lower_serafim = mu_serafim + k*sigma_serafim, mu_serafim - k*sigma_serafim
upper_labse,  lower_labse  = mu_labse  + k*sigma_labse,  mu_labse  - k*sigma_labse

contratos = contratos.with_columns([
    pl.when(
        (pl.col("flag_1011_anom") == 0) &
        (pl.col("dist_serafim_obj_tipo").is_between(lower_serafim, upper_serafim).not_())
    ).then(pl.lit(1)).otherwise(pl.lit(0)).alias("flag_1011_serafim"),

    pl.when(
        (pl.col("flag_1011_anom") == 0) &
        (pl.col("dist_labse_obj_tipo").is_between(lower_labse, upper_labse).not_())
    ).then(pl.lit(1)).otherwise(pl.lit(0)).alias("flag_1011_labse"),
])

contratos = contratos.with_columns(
    pl.when(
        (pl.col("flag_1011_serafim")==1) & (pl.col("flag_1011_labse")==1) & (pl.col("flag_1011_anom")==0)
    ).then(pl.lit(1)).otherwise(pl.lit(0)).alias("flag_1011_dist")
)


contratos.select(["N.º Procedimento (ID BASE)", "N.º Contrato", "flag_1011_anom", "flag_1011_dist", "objeto_len"]).write_ipc("../../data/indicators/1011_a.arrow")
