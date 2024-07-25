import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt


def getAcidentes():
    return pd.read_csv('./data/acidentes.csv', delimiter=',', encoding='ISO-8859-1', low_memory=False)


def getAcidentesPorRegiaoTotal():
    # Carregar os dados
    dataSetAcidentes = getAcidentes()

    # Filtrar acidentes com regiões válidas (não nulas e não vazias)
    dataSetAcidentes = dataSetAcidentes[dataSetAcidentes['DESC_REGIONAL'].notna() & (dataSetAcidentes['DESC_REGIONAL'].str.strip() != '')]

    # Extrair a data e a região
    dataSetAcidentes['DATA_BOLETIM'] = pd.to_datetime(dataSetAcidentes['DATA HORA_BOLETIM'], format='%d/%m/%Y %H:%M', errors='coerce')
    dataSetAcidentes['ANO'] = dataSetAcidentes['DATA_BOLETIM'].dt.year
    dataSetAcidentes = dataSetAcidentes[(dataSetAcidentes['ANO'] >= 2012) & (dataSetAcidentes['ANO'] <= 2022)]

    # Gráfico de barras: Regiões com mais acidentes
    acidentes_por_regiao = dataSetAcidentes['DESC_REGIONAL'].value_counts()

    plt.figure(figsize=(12, 8))
    acidentes_por_regiao.plot(kind='bar')
    plt.xlabel('Região')
    plt.ylabel('Quantidade de Acidentes')
    plt.title('Quantidade de Acidentes por Região')
    plt.show()


def getAcidentesPorRegiaoAno():
    # Carregar os dados
    dataSetAcidentes = getAcidentes()

    # Filtrar acidentes com regiões válidas (não nulas e não vazias)
    dataSetAcidentes = dataSetAcidentes[dataSetAcidentes['DESC_REGIONAL'].notna() & (dataSetAcidentes['DESC_REGIONAL'].str.strip() != '')]

    # Extrair a data e a região
    dataSetAcidentes['DATA_BOLETIM'] = pd.to_datetime(dataSetAcidentes['DATA HORA_BOLETIM'], format='%d/%m/%Y %H:%M', errors='coerce')
    dataSetAcidentes['ANO'] = dataSetAcidentes['DATA_BOLETIM'].dt.year
    dataSetAcidentes = dataSetAcidentes[(dataSetAcidentes['ANO'] >= 2012) & (dataSetAcidentes['ANO'] <= 2022)]

    # Gráfico de linhas: Quantidade de acidentes por ano para cada região
    acidentes_por_ano_regiao = dataSetAcidentes.groupby(['ANO', 'DESC_REGIONAL']).size().unstack(fill_value=0)

    plt.figure(figsize=(14, 10))
    for regiao in acidentes_por_ano_regiao.columns:
        plt.plot(acidentes_por_ano_regiao.index, acidentes_por_ano_regiao[regiao], marker='o', label=regiao)

    plt.xlabel('Ano')
    plt.ylabel('Quantidade de Acidentes')
    plt.title('Quantidade de Acidentes por Ano e Região (2012-2022)')
    plt.legend(title='Região', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()



def main():
    getAcidentesPorRegiaoAno();
    

if __name__ == "__main__":
    main()