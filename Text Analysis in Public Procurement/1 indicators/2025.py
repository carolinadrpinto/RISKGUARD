import polars as pl
import numpy as np
import pandas as pd

from utilities.schema import contracts_schema, lots_schema, modifications_schema


SEED=42
np.random.seed(SEED)



# fazer load do dataset contratos, só das colunas relevantes e transformar devidamente as datas em formatos adequados
# porque vêm em formato estranho
# "'2017-12-28 00:00:00.000'
# ainda tenho de pensar como vou fazer o preprocessamento aqui

contracts = pl.read_csv("../../data/impic_data/contratos.csv", separator=";", schema_overrides=contracts_schema(),
    columns=[
        "Tipo(s) de contrato", "Causas das alterações ao prazo de execução do contrato", "Preço Contratual (€)",
        "Causas das alterações no valor do contrato", "Causas das alterações ao prazo de execução da obra",
        "Causas das alterações no valor da obra", "Informação sobre Trabalhos Complementares",
        "N.º Procedimento (ID BASE)", "N.º Contrato", "Contratação Excluída", "Prorrogação de Prazo", "Data Celebração"
        ], 
    null_values=["NULL"]) \
    .with_columns(pl.col('Data Celebração').replace("NULL", None).str.split(" ").list.first().str.replace("'", "").str.to_date("%F")) \
    .unique(subset=["N.º Contrato", "N.º Procedimento (ID BASE)"]) \
    .filter(pl.col("Contratação Excluída") == False)



# preprocess contracts
contratos_cleaned = pl.read_ipc("../../data/clean_up_data/contracts_cleanedup.arrow")

# preprocess contracts
contratos_prepro = contracts.join(contratos_cleaned,
                                  how="anti", 
                                  on = ["N.º Procedimento (ID BASE)", "N.º Contrato"], 
                                  coalesce=True)


anexo_xii = pl.read_csv("../../data/impic_data/anexo_xii.csv", separator=";", schema_overrides=modifications_schema(),
    columns=[
       "N.º Procedimento (ID BASE)", "N.º Contrato",
       "Data da Publicação da MC", "Data da Modificação",
       "Fundamentação da MC", "Tipo do Ato", "Nº do Lote",
       "Preço Contratual Original", "Novo Preço Contratual",
       "Objeto da subcontratação",
       "Subcontratação não ultrapassa valor total superior a 75 % do preço contratual (383.º nº2 do CCP)"
    ], 
    null_values=["NULL"]) \
    .with_columns(
        pl.col("Data da Publicação da MC").replace("NULL", None).str.split(" ").list.first().str.replace("'", "").str.to_date("%F"),
        pl.col("Data da Modificação").replace("NULL", None).str.split(" ").list.first().str.replace("'", "").str.to_date("%F"),
        pl.col("Data da Publicação da MC").replace("NULL", None).str.split(" ").list.get(1).str.replace("'", "").str.strptime(pl.Time, "%H:%M:%S%.f").alias("Hora da Publicação")
    )

anexo_xii_cleaned = pl.read_ipc("../../data/clean_up_data/modifications_cleanedup.arrow")

# preprocess contracts
anexo_xii_prepro = anexo_xii.join(anexo_xii_cleaned,
                                  how="anti", 
                                  on = ["N.º Procedimento (ID BASE)", "N.º Contrato"], 
                                  coalesce=True)


lotes = pl.read_csv("../../data/impic_data/lotes.csv", separator=";", schema_overrides=lots_schema(),
    columns=[
       "IDAliasProc", "ContractID",
       "Preço Contratual", "Número de Ordem do Lote",
       "Valor do Lote"
    ], 
    null_values=["NULL"])


lotes_cleaned = pl.read_ipc("../../data/clean_up_data/lots_cleanedup.arrow")

# preprocess contracts
lotes_prepro = lotes.join(lotes_cleaned,
                                  how="anti", 
                                  on = ["IDAliasProc", "ContractID"], 
                                  coalesce=True)



mods_funds_per_contract = anexo_xii_prepro.group_by("N.º Procedimento (ID BASE)", "N.º Contrato", "Nº do Lote").agg(pl.col("Fundamentação da MC").unique(), pl.len().alias("num_mods")) \
                                    .with_columns(pl.col("Fundamentação da MC").list.len().alias("num_funds")) \
                                    .select(["N.º Procedimento (ID BASE)", "N.º Contrato", "Nº do Lote", "num_funds", "num_mods"])

# dataset with all the modifications informations

# - Num modifications per contract
# - Num diff funds per contract
modifications_data = anexo_xii_prepro.join(mods_funds_per_contract, 
                                    how="left", 
                                    on=["N.º Procedimento (ID BASE)", "N.º Contrato", "Nº do Lote"], coalesce=True)


contratos_lotes = contratos_prepro.join(lotes_prepro, how="left", 
                                 left_on=["N.º Procedimento (ID BASE)", "N.º Contrato"], 
                                 right_on=["IDAliasProc", "ContractID"], coalesce=True)


# ver o relatório mais recente por contrato -> cada linha é um contrato e o respetivo lote e a data mais recente da mod
aux_data = anexo_xii_prepro.group_by("N.º Procedimento (ID BASE)", "N.º Contrato", "Nº do Lote") \
    .agg(pl.max("Data da Publicação da MC"))

# relatórios a mais no anexo -> todos os relaórios que não têm a data mais recente
anti = anexo_xii_prepro.join(aux_data, how="anti", on=["N.º Procedimento (ID BASE)", "N.º Contrato", "Nº do Lote", "Data da Publicação da MC"])

# todos os relatórios relevantes -> relatórios que têm a data mais recente
mod_aux = anexo_xii_prepro.join(anti, how="anti", on=["N.º Procedimento (ID BASE)", "N.º Contrato", "Nº do Lote", "Data da Publicação da MC"])

# tirar o publicado mais tardiamente (ver pela hora mais tarde)
mod_dia_hora_pub = mod_aux.group_by("N.º Procedimento (ID BASE)", "N.º Contrato", "Nº do Lote", "Data da Publicação da MC") \
    .agg(pl.max("Hora da Publicação")).with_columns(pl.lit(0).alias("last_mod")) # se o relatório foi a last mod então tem um zero

modifications_data = modifications_data.join(mod_dia_hora_pub, how="left", on=["N.º Procedimento (ID BASE)", "N.º Contrato", "Nº do Lote", "Data da Publicação da MC", "Hora da Publicação"], coalesce=True) \
    .with_columns(pl.col("last_mod").fill_null(1))


def indicator_2025(contracts_lotes: pl.DataFrame, modifications_data: pl.DataFrame):

    "Check the compliance of justifications for contract modifications."

    "If the justification is not about 'prazo' or 'preço' or 'tipo(s) de contrato' the the function is not signlazing incoherences"

    "For the contracts with more than one modification, see if the scopes of the modifications are different. If so, sinalize that"

    tipos_contrato = contratos_lotes.with_columns(pl.col("Tipo(s) de contrato").str.to_lowercase()).select(pl.col("Tipo(s) de contrato").drop_nulls().unique()).to_series().to_list()


    return contracts_lotes.join(modifications_data, how="left", left_on=["N.º Procedimento (ID BASE)", "N.º Contrato", "Número de Ordem do Lote"], 
                         right_on = ["N.º Procedimento (ID BASE)", "N.º Contrato", "Nº do Lote"],
                         coalesce=True).with_columns((pl.col("Novo Preço Contratual")-pl.col("Preço Contratual Original")).alias("price_change"), 
                                                     pl.col("Fundamentação da MC").str.to_lowercase(),
                                                     pl.col("Tipo(s) de contrato").str.to_lowercase())\
                                .with_columns(
                                    pl.when(
                                        # Fundamentaçãao da MC menciona preço - verificar se de facto houve alteração de preço
                                        ((pl.col("Fundamentação da MC").str.contains("preço?[s]? | financeir[oa]?[s]?")) 
                                            & ((pl.col("price_change").is_nan()) 
                                            | (pl.col("price_change")==0)))

                                        # Fundamentação da MC menciona prorrogação de prazo - verificar se houve alteração de prazo
                                        | ((pl.col("Fundamentação da MC").str.contains("prorrogação ?(de|do)? prazo"))
                                            & ((pl.col("Prorrogação de Prazo").is_null())
                                            | (pl.col("Prorrogação de Prazo")=="0")))

                                        # Fundamentação da MC menciona algum tipo de contrato - verificar se esse é o tipo de contrato correspondente
                                        | ((pl.col("Fundamentação da MC").str.contains_any(tipos_contrato)) 
                                            & (~(pl.col("Fundamentação da MC").str.contains(pl.col("Tipo(s) de contrato")))))
                                        ) \
                                .then(pl.lit(1)).otherwise(pl.lit(0)).alias("indicator_fund")
                                ) \
                            .select(["N.º Procedimento (ID BASE)", "N.º Contrato", "Número de Ordem do Lote", "Fundamentação da MC", "Prorrogação de Prazo", "price_change", "Tipo(s) de contrato", "indicator_fund", "num_funds", "num_mods", "last_mod", "Data da Modificação", "Data Celebração"])
    

ind_2025 = indicator_2025(contratos_lotes, modifications_data)
ind_2025.write_ipc("../../data/indicators/2025.arrow")
#indicator_2014_2023(contracts_just).write_ipc("../data/indicators/2014_2023.arrow")