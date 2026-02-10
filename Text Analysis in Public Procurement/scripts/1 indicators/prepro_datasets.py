import polars as pl

from utilities.prepro_fun import*

from utilities.schema import *

def store_cleanup_data(cleanup_data: pl.DataFrame, cleanup_name: str):
    cleanup_data.write_ipc("../cleanup_data/" + cleanup_name + ".arrow")


if __name__ == "__main__":
    
    # Loading contracts
    
    contracts = pl.read_csv(
        "../../data/impic_data/contratos.csv", 
        schema_overrides=contracts_schema(),
        separator=";",
        columns=[
            "N.º Contrato", "N.º Procedimento (ID BASE)", "Entidade(s) Adjudicatária(s) - NIF", 
            "Entidade(s) Adjudicante(s) - NIF", "Contratação Excluída", "Preço Contratual (€)", "Tipo(s) de contrato",
            "Data da decisão adjudicação", "Data Celebração", "Preço BASE (€)"
            ]
        )  
    
    # Get null counts
    contracts_null_counts = contracts.null_count()

    contracts = contracts \
            .with_columns(
            pl.col("Data Celebração").replace("NULL", None).str.split(" ").list.first().str.replace("'", "").str.to_date("%F"),
            pl.col("Data da decisão adjudicação").replace("NULL", None).str.split(" ").list.first().str.replace("'", "").str.to_date("%F"),
        ).filter(pl.col("Contratação Excluída") == False)
    

    contracts_cleaned = contracts_cleanup(contracts)
    contracts_cleaned.write_ipc("../../data/clean_up_data/contracts_cleanedup.arrow")
    

    modifications_data = pl.read_csv(
        "../../data/impic_data/anexo_xii.csv",
        separator=";",
        null_values=["NULL", ""],
        schema_overrides=modifications_schema(),
        columns=[
            "Preço Contratual Original", "Novo Preço Contratual", "Data da Modificação",
            "N.º Procedimento (ID BASE)", "N.º Contrato", "Nº do Lote",
            "Subcontratação não ultrapassa valor total superior a 75 % do preço contratual (383.º nº2 do CCP)"
        ]
    )
    
    # Get null counts
    modifications_null_counts = modifications_data.null_count()
    modifications_cleaned = modifications_data_cleanup(modifications_data)
    modifications_cleaned.write_ipc("../../data/clean_up_data/modifications_cleanedup.arrow")


    lots = pl.read_csv(
            "../../data/impic_data/lotes.csv", 
            separator=";", 
            schema_overrides=lots_schema(),
            columns=["IDAliasProc", "ContractID", "Número de Ordem do Lote", "Preço Contratual", "Valor do Lote"]
        ) 
    
    # Get null counts
    lots_null_counts = lots.null_count()
    lots_cleaned = lots_cleanup(lots)
    lots_cleaned.write_ipc("../../data/clean_up_data/lots_cleanedup.arrow")


    anexo_xiv = pl.read_csv(
            "../../data/impic_data/anexo_xiv.csv",
            separator=';',
            null_values=["NULL"],
            schema_overrides=anexo_xiv_schema(),
            columns=['N.º Procedimento (ID BASE)', 'ID Contrato',
                     'Preço Total Efetivo', 'Preço unitário (caso aplicável)',
                     'Fundamentação da eficácia retroativa ao contrato']
    )
    
    anexo_xiv_null_counts = anexo_xiv.null_count()
    anexo_xiv_cleaned = anexo_xiv_data_cleanup(anexo_xiv)
    anexo_xiv_cleaned.write_ipc("../../data/clean_up_data/anexo_xiv_cleanedup.arrow")


    
    bid_level_data = pl.read_csv(
            "../../data/impic_data/anexo_vii.csv",
            separator=";",
            schema_overrides=bid_schema(),
        ) 
    
    bids_cleaned = bid_level_data_cleanup(bid_level_data)
    bids_cleaned.write_ipc("../../data/clean_up_data/bids_cleanedup.arrow")

    

