import os

import pandas as pd
import polars as pl
import numpy as np

SEED=42
np.random.seed(SEED)

from utilities.funcoes import prepro_obj, generate_embeddings_serafim335, generate_embeddings_labse
from utilities.schema import contracts_schema

# # COMENTAR SE NÃO FOR NO SERVIDOR
# from dotenv import load_dotenv
# load_dotenv()


#contratos = pd.read_csv(os.getenv('DATA_ROOT') + '/contratos.csv', sep=';')

contratos = pl.read_csv(
    "../../data/impic_data/contratos.csv",
    separator=";",
    schema_overrides=contracts_schema(),
    columns=[
        "Tipo(s) de contrato", "Objeto", "N.º Procedimento (ID BASE)",
        "N.º Contrato", "Contratação Excluída", "Ajuste Direto Simplificado", "Tipo de procedimento", "Data Celebração"
    ],
    null_values=["NULL"]
) \
    .unique(subset=["N.º Contrato", "N.º Procedimento (ID BASE)"]) \
    .filter((pl.col("Contratação Excluída") == False) & (pl.col("Ajuste Direto Simplificado") == False ))

contratos_cleaned = pl.read_ipc("../../data/clean_up_data/contracts_cleanedup.arrow")

# ATENÇÃO AQUI

# fazer embeddings para todos exceto para contratação excluída e para ajuste direto simplificado
contratos_prepro = contratos.join(
    contratos_cleaned,
    how="anti",
    on=["N.º Procedimento (ID BASE)", "N.º Contrato"],
    coalesce=True
)

with open('pt-stopwords.txt', "r", encoding="utf-8") as f:
    stopwords = {line.strip().lower() for line in f if line.strip()}
stopwords = stopwords.union({'n.º'})

contratos_cleaned_prepro = prepro_obj(contratos_prepro, stopwords)


# GENERATE EMBEDDINGS CODE

embedding_models = ["serafim", "labse"]
embedding_functions = [
    generate_embeddings_serafim335,
    generate_embeddings_labse,
]

cols = ["Objeto_LIMPO", "Tipo(s) de contrato_LIMPO"]

# ✅ chaves para garantir associação embeddings <-> contrato
id_cols = ["N.º Procedimento (ID BASE)", "N.º Contrato"]

for embedding_model, function in zip(embedding_models, embedding_functions):
    os.makedirs(f"../../data/embeddings/{embedding_model}", exist_ok=True)
    print(f"Running {embedding_model} ...")
    for i, col in enumerate(cols):
        kwargs = {"df": contratos_cleaned_prepro, "text_column": col}
        if embedding_model in ["serafim", "labse"]:
            kwargs["batch_size"] = 32

        embeddings = function(**kwargs)

        df_dict = {f"{col}_dim{j}": embeddings[:, j].tolist() for j in range(embeddings.shape[1])}
        embeddings_only_df = pl.DataFrame(df_dict)

        # ✅ guarda IDs + embeddings no mesmo parquet (independente de ordem)
        embeddings_df = pl.concat(
            [contratos_cleaned_prepro.select(id_cols), embeddings_only_df],
            how="horizontal"
        )

        file_path = os.path.join(f"../../data/embeddings/{embedding_model}", f"{col}.parquet")
        embeddings_df.write_parquet(file_path)
        print(f"Done col {i+1}/{len(cols)} .")

contratos_cleaned_prepro.write_parquet("../../data/contratos_cleaned_prepro.parquet")
