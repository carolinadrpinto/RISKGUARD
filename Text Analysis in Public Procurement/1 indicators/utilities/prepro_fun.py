import polars as pl
from datetime import datetime



def contracts_cleanup(contracts_table: pl.DataFrame, remove_year_before: int = 2017) -> pl.DataFrame:
    """Define cleanup filters for contracts table

    Steps performed:
        1. Remove contracts with negative prices
        2. Remove contracts with years before ``remove_year_before`` (Defaults to 2017)
        3. Remove contracts without NIF for public entity and no contract signing date
        4. Remove contracts with future dates

    Args:
        contracts_table (pl.DataFrame): The contracts table (contratos)
        remove_year_before (int): The year before which contracts should be removed

    Returns:
        pl.DataFrame: The procedure id and contract id of the removed contracts
    """
    return contracts_table \
        .filter(
            (pl.col("Preço Contratual (€)") <= 0) |
            (pl.col("Preço BASE (€)") < 0) |
            (pl.col("Data da decisão adjudicação").dt.year() <= remove_year_before) |
            (pl.col("Data da decisão adjudicação") > datetime.utcnow()) |
            (pl.col("Data Celebração").dt.year() <= remove_year_before) |
            (pl.col("Data Celebração").dt.year() > datetime.now())
        ) \
        .select("N.º Procedimento (ID BASE)", "N.º Contrato")


def lots_cleanup(lots_table: pl.DataFrame) -> pl.DataFrame:
    """Define cleanup filters for lots table

    Args:
        lots_table (pl.DataFrame): The lots table (lotes)

    Returns:
        pl.DataFrame: The procedure id and contract id of the lots to be removed
    """
    return lots_table \
        .filter(
            (pl.col("Preço Contratual") < 0) |
            (pl.col("IDAliasProc").is_null()) |
            (pl.col("ContractID").is_null()) |
            (pl.col("Valor do Lote") < 0)
        ) \
        .select("IDAliasProc", "ContractID", "Número de Ordem do Lote")


def modifications_data_cleanup(modifications_data: pl.DataFrame) -> pl.DataFrame:
    """Define cleanup filters for 

    Args:
        modifications_data (pl.DataFrame): _description_

    Returns:
        pl.DataFrame: _description_
    """
    return modifications_data \
        .filter(pl.col("Novo Preço Contratual") < 0) \
        .select(["N.º Procedimento (ID BASE)", "N.º Contrato", "Nº do Lote"])



def anexo_xiv_data_cleanup(anexo_xiv: pl.DataFrame) -> pl.DataFrame:
    """Define cleanup filters for anexo_ii table

    Args:
        anexo_xiv (pl.DataFrame): The anexo_xiv table (lotes)

    Returns:
        pl.DataFrame: The procedure id and contract id of the anexo_xiv rows to be removed
    """
    return anexo_xiv \
        .filter(pl.col('Preço Total Efetivo') < 0)\
        .select(['N.º Procedimento (ID BASE)', 'ID Contrato'])


def bid_level_data_cleanup(bid_level_data: pl.DataFrame) -> pl.DataFrame:
    """Performs a cleanup on the bid level data (Annex VII)

    This function checks and returns the procedure ids of bids that
        1. Have a negative base price
        2. Have a negative proposal value  

    Returns:
        pl.DataFrame: A list of procedure ids
    """

    return bid_level_data \
        .filter(
        (pl.col("Preço base do lote em causa") < 0) |
        (pl.col("Valor da proposta") < 0)
     ) \
    .select(["Dados de base do procedimento", "Número de ordem do lote em causa"])
        