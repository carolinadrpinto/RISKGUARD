import os

import pandas as pd
import polars as pl
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

SEED=42
np.random.seed(SEED)

from utilities.funcoes import get_covid_legal_frameworks, preprocess_text
from utilities.schema import contracts_schema, procedimentos_schema

# already preprocessed

contratos = pl.read_parquet(
    "../../data/contratos_cleaned_prepro.parquet",
    columns=[
        "N.º Procedimento (ID BASE)", "N.º Contrato", "Tipo(s) de contrato_LIMPO", "Objeto_LIMPO", "Objeto_LIMPO_2", "Objeto", "Tipo de procedimento"
    ]
    )


procedimentos = pl.read_csv("../../data/impic_data/procedimentos.csv",
                            separator=";",
                            schema_overrides=procedimentos_schema(),
                            columns=[
                                "ContractingProcedureAliasID", "Regime de Contratação"
                                     ],
                            null_values=["NULL"]) \
                            .unique(subset="ContractingProcedureAliasID")



contratos_raw = pl.read_csv("../../data/impic_data/contratos.csv", separator=";", schema_overrides=contracts_schema(),
    columns=['N.º Procedimento (ID BASE)', 'N.º Contrato', 'Data da decisão adjudicação', 
             'Data Celebração', "Contratação Excluída", "Tipo de procedimento", "Medidas Especiais",
              "Ao abrigo dos critérios materiais", 'Preço BASE (€)', 'Preço Contratual (€)'], 
    null_values=["NULL"]) \
    .with_columns(
        pl.col("Data da decisão adjudicação").replace("NULL", None).str.split(" ").list.first().str.replace("'", "").str.to_date("%F"),
        pl.col('Data Celebração').replace("NULL", None).str.split(" ").list.first().str.replace("'", "").str.to_date("%F"),
    ) \
    .with_columns(
        pl.col("Data Celebração").dt.year().alias("contract_year")
    ) \
    .join(procedimentos, how="left", left_on="N.º Procedimento (ID BASE)", right_on="ContractingProcedureAliasID", coalesce=True) \
    .filter(pl.col("Contratação Excluída") == False) \
    .filter(pl.col("Tipo de procedimento").is_in(["Concurso público", "Concurso limitado por prévia qualificação"])) \
    .filter(pl.col("Medidas Especiais").is_null()) \
    .filter(~pl.col("Regime de Contratação").str.contains_any(get_covid_legal_frameworks())) \
    .unique(subset=['N.º Procedimento (ID BASE)', 'N.º Contrato']).select(['N.º Procedimento (ID BASE)', 'N.º Contrato', "contract_year"])

contratos = contratos.join(contratos_raw, how="inner", on=['N.º Procedimento (ID BASE)', 'N.º Contrato'], coalesce=True)

contratos = contratos.with_columns(
    pl.col("Objeto_LIMPO").map_elements(preprocess_text, return_dtype=pl.Utf8).alias("Objeto_LIMPO_Final")
)


texts = contratos.select("Objeto_LIMPO").to_series().to_list()

vectorizer = TfidfVectorizer(min_df=5)
X = vectorizer.fit_transform(texts)

tfidf_pl = pl.DataFrame(
    X.toarray(),
    schema=vectorizer.get_feature_names_out().tolist()
)


result = contratos.select(['N.º Procedimento (ID BASE)', 'N.º Contrato']).hstack(tfidf_pl)


result.write_parquet("../../data/embeddings/tdidf/tfidf_objeto.parquet")
