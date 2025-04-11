import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns


st.set_page_config(page_title="Finance Analysis", layout="wide")
st.image("logo.svg", width=150)

df = pd.read_csv("dataset_finance.csv")


import plotly.express as px
import pandas as pd
import streamlit as st

# Top 10 produtos com mais reclamações
st.subheader("📌 Top 10 Produtos Mais Reclamados")
top_produtos = df["Product"].value_counts().head(10)
st.write(top_produtos)

# Filtrar o dataframe para conter só os top 10 produtos
produtos_filtrados = df[df["Product"].isin(top_produtos.index)]

# Agrupar para ver os principais motivos por produto
top_issues_por_produto = (
    produtos_filtrados
    .groupby(["Product", "Issue"])
    .size()
    .reset_index(name="Quantidade")
)


top_issues_por_produto = (
    top_issues_por_produto
    .sort_values(["Product", "Quantidade"], ascending=[True, False])
    .groupby("Product")
    .head(5)
)


st.markdown("""
    <h4 style="color: #2aa0d3;">Quais os três produtos mais reclamados?</h4>
    <h5>Isso indica que há um padrão de insatisfação com o processo de endividamento, seja pelo acesso ao crédito, cobrança indevida ou dificuldades no gerenciamento de contas bancárias. Esses dados sugerem a necessidade de maior transparência, comunicação e suporte ao cliente por parte das instituições financeiras.</h5>
    """, unsafe_allow_html=True)


#----------------------------
st.subheader("📊 Matriz de Correlação")

# Seleciona apenas as colunas relevantes
corr_cols = ["Resolution time(in days)", "Consumer disputed"]
corr_matrix = df[corr_cols].corr()

# Cria o gráfico
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap="YlOrBr", fmt=".2f", ax=ax)
ax.set_title("Correlação entre Tempo de Resolução e Contestação do Consumidor")

# Mostra no Streamlit
st.pyplot(fig)

st.markdown("""
    <h4 style="color: #2aa0d3;">3. Existe correlação entre o tempo de resposta e a contestação pelo consumidor?</h4>
    <h5>O indice de corelação é negativo porém muito baixo, nao podendo inferir se há uma correlação, positivba ou negativa</h5>
    """, unsafe_allow_html=True)