# ATENÇÃO À PASTA E AO CAMINHO


import polars as pl
import numpy as np


SEED=42
np.random.seed(SEED)


from Contratos.scripts.scripts_scrape.utilities.schema import anexo_xiv_schema, contracts_schema

# fazer load do dataset contratos, só das colunas relevantes e transformar devidamente as datas em formatos adequados
# porque vêm em formato estranho
# "'2017-12-28 00:00:00.000'
# ainda tenho de pensar como vou fazer o preprocessamento aqui

contracts = pl.read_csv("../../data/impic_data/contratos.csv", separator=";", schema_overrides=contracts_schema(),
    columns=[
        "Data de fecho do contrato", "Data da receção provisória (art.º 395.º do CCP)",
        "Tipo(s) de contrato", "Data da decisão adjudicação", "N.º Procedimento (ID BASE)", 
        "Data Decisão Contratar", "Data Celebração", "N.º Contrato", "Contratação Excluída"
    ], 
    null_values=["NULL"]) \
    .with_columns(
        pl.col("Data de fecho do contrato").replace("NULL", None).str.split(" ").list.first().str.replace("'", "").str.to_date("%F"),
        pl.col("Data da decisão adjudicação").replace("NULL", None).str.split(" ").list.first().str.replace("'", "").str.to_date("%F"),
        pl.col("Data Decisão Contratar").replace("NULL", None).str.split(" ").list.first().str.replace("'", "").str.to_date("%F"),
        pl.col('Data Celebração').replace("NULL", None).str.split(" ").list.first().str.replace("'", "").str.to_date("%F"),
        pl.col("Data da receção provisória (art.º 395.º do CCP)").replace("NULL", None).str.split(" ").list.first().str.replace("'", "").str.to_date("%F")
    ) \
    .filter(pl.col("Contratação Excluída") == False)



contratos_cleaned = pl.read_ipc("../../data/clean_up_data/contracts_cleanedup.arrow")

# preprocess contracts
contratos_prepro = contracts.join(contratos_cleaned,
                                  how="anti", 
                                  on = ["N.º Procedimento (ID BASE)", "N.º Contrato"], 
                                  coalesce=True)



anexo_xiv = pl.read_csv("../../data/impic_data/anexo_xiv.csv", separator=";", schema_overrides=anexo_xiv_schema(),
    columns=[
        "N.º Procedimento (ID BASE)", "ID Contrato",
        "Fundamentação da eficácia retroativa ao contrato",
        "Data de Fecho do Contrato", "Data de Fecho do Contrato Física"
    ], 
    null_values=["NULL"]) \
    .with_columns(
        pl.col("Data de Fecho do Contrato").replace("NULL", None).str.split(" ").list.first().str.replace("'", "").str.to_date("%F"),
        pl.col("Data de Fecho do Contrato Física").replace("NULL", None).str.split(" ").list.first().str.replace("'", "").str.to_date("%F"),
    )


anexo_xiv_cleaned = pl.read_ipc("../../data/clean_up_data/anexo_xiv_cleanedup.arrow")

# preprocess contracts
anexo_xiv_prepro = anexo_xiv.join(anexo_xiv_cleaned,
                                  how="anti", 
                                  on = ["N.º Procedimento (ID BASE)", "ID Contrato"], 
                                  coalesce=True)

# join contracts and anexo_xi
contracts_just = contratos_prepro.join(anexo_xiv_prepro, how="left", left_on=["N.º Procedimento (ID BASE)", "N.º Contrato"], right_on=["N.º Procedimento (ID BASE)", "ID Contrato"], coalesce=True)

def indicator_2014_2023(contratos_just: pl.DataFrame) -> pl.DataFrame:
    "Completa os indicadores 1014 e 1023 através da análise da coluna de texto: "
    " Analisa a coluna 'Fundamentação da eficácia retroativa ao contrato'"

    contratos_just_return = contratos_just.with_columns(
        [
        pl.when(
                pl.col("Data de fecho do contrato") < pl.col("Data da decisão adjudicação"
                                                             )).then(pl.lit(0)).otherwise(pl.lit(1)).alias("1014_a"),
        pl.when(
                ((pl.col("Tipo(s) de contrato") == "Empreitadas de obras públicas") &
                 # data em que se deram concluídos os trabalhos
                (pl.col("Data da receção provisória (art.º 395.º do CCP)") < pl.col("Data da decisão adjudicação"))
                )
        ).then(pl.lit(0)).otherwise(pl.lit(1)).alias("1014_b"),
        pl.when(
                pl.col("Data Celebração") < pl.col("Data da decisão adjudicação")
        ).then(pl.lit(0)).otherwise(pl.lit(1)).alias("1023")
        ]
        ).select("N.º Procedimento (ID BASE)", "N.º Contrato", "1014_a", "1014_b", "1023", "Fundamentação da eficácia retroativa ao contrato")
    
    return contratos_just_return.with_columns(
        pl.when(
            # Se houver eficácia retroativa do contrato, perceber se é mencionado o artigo em questão. 
            # Caso contrário, sinalizar o contrato como possivelmente problemático
            (((pl.col("1014_a")==0) | (pl.col("1014_b")==0) | (pl.col("1023")==0)) & 
            (pl.col("Fundamentação da eficácia retroativa ao contrato").str.contains("([Aa]rtigo\s+(?:(?:n[.ºo]?)|n\.º|número)?\s*287)")))
            ).then(pl.lit(0)).otherwise(pl.lit(1)).alias("2014_23")

    ).select("N.º Procedimento (ID BASE)", "N.º Contrato", "1014_a", "1014_b", "1023", "2014_23")

ind_2014_2023 = indicator_2014_2023(contracts_just)
ind_2014_2023.write_ipc("../../data/indicators/2014_2023.arrow")
#indicator_2014_2023(contracts_just).write_ipc("../data/indicators/2014_2023.arrow")