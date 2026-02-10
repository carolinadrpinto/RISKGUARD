import polars as pl
from typing import Dict

def contracts_schema() -> Dict[str, pl.DataType]:
    return {
        "N.º Contrato": pl.Utf8,
        "N.º Procedimento (ID BASE)": pl.Utf8,
        "Objeto": pl.Utf8,
        "Entidade(s) Adjudicatária(s) - NIF": pl.Utf8,
        "Entidade(s) Adjudicante(s) - NIF": pl.Utf8,
        "Tipo(s) de contrato": pl.Utf8,
        "Preço Contratual (€)": pl.Float64,
        "Valor estimado do(s) contrato(s) (€) (s/IVA)": pl.Float64,
        "Preço BASE (€)": pl.Float64,
        "Prorrogação de Prazo": pl.Utf8
    }

def procedimentos_schema() -> Dict[str, pl.DataType]:
    return {
        "ContractingProcedureAliasID": pl.Utf8, 
        "Número do Anúncio (Nº/Ano)": pl.Utf8, 
        "Descrição/designação do procedimento": pl.Utf8, 
        "URL Peças do Procedimento": pl.Utf8
    }
    
def lots_schema() -> Dict[str, pl.DataType]:
    return {
        "Número de Ordem do Lote": pl.Utf8,
        "IDAliasProc": pl.Utf8,
        "ContractID": pl.Utf8,
        "Preço Contratual": pl.Float64,
        "Valor do Lote": pl.Float64
    }

def modifications_schema() -> Dict[str, pl.DataType]:
    return {
        "N.º Procedimento (ID BASE)": pl.Utf8,
        "N.º Contrato": pl.Utf8,
        "Fundamentação da MC": pl.Utf8,
        "Tipo do Ato": pl.Utf8,
        "Nº do Lote": pl.Utf8,
        "Preço Contratual Original": pl.Float64,
        "Novo Preço Contratual": pl.Float64,
    }

def anexo_xiv_schema() -> Dict[str, pl.DataType]:
    return{
        "N.º Procedimento (ID BASE)": pl.Utf8,
        'ID Contrato': pl.Utf8,
        'Preço Total Efetivo': pl.Float64, 
        'Preço unitário (caso aplicável)': pl.Float64,
        'Fundamentação da eficácia retroativa ao contrato': pl.Utf8
    }


def bid_schema() -> Dict[str, pl.DataType]:
    return{
        "Número de ordem do lote em causa": pl.Utf8, 
        "Código da proposta": pl.Utf8,
        "Dados de base do procedimento": pl.Utf8 

    }
