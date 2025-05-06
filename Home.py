import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns


st.set_page_config(page_title="Finance Analisys", layout="wide")

#--------------------------------------------------

st.image("logo.svg", width=150)

df = pd.read_csv("dataset_finance.csv")

df.columns = df.columns.str.strip()

st.title("Análise de Reclamações de Consumidores da Amazon")
st.write("""
Esta análise explora reclamações de consumidores sobre produtos financeiros nos Estados Unidos. A base de dados inclui informações sobre empresas, tipos de produto, motivos de insatisfação, localização e tempo de resolução. O foco é identificar padrões nos sentimentos dos clientes e propor melhorias que otimizem a experiência do consumidor, com atenção especial a empresas como a Amazon, se presentes no conjunto de dados.
""")

st.markdown("<h1 style='color: #2aa0d3;'>1️⃣ Apresentação das variáveis</h1>", unsafe_allow_html=True)
st.subheader("Amostra dos Dados")
st.write(df.head())


dados_tipos = {
    "Variável": [
        "ID", "Company", "Product", "Issue", "State", "Submitted via",
        "Date received", "Date resolved", "Timely Reponse", "Consumer disputed?",
        "State Name", "Resolution Time(In days)", "Year", "QTR"
    ],
    "Tipo de Variável": [
        "Identificador", "Nominal", "Nominal", "Nominal", "Nominal", "Nominal",
        "Data", "Data", "Data", "Nominal",
        "Nominal", "Contínua", "Data", "Contínua"
    ]
}

#--------------------------------------------------

df_tipos = pd.DataFrame(dados_tipos)

# Exibindo no Streamlit
st.subheader("Classificação das Variáveis")
st.dataframe(df_tipos)


st.markdown("<h1 style='color: #2aa0d3;'>2️⃣Questionamentos e Hipóteses</h1>", unsafe_allow_html=True)

st.write("""
1. Qual o tempo médio de resolução das reclamações?
2. Quais os três estados com mais reclamações?
3. Existe correlação entre o estado e o tempo de reclamação?
4. Qual o intervalo de confiança que uma reclamação será respondida dentro do tempo médio?
5. H0 Estados com mais reclamações (State) têm tempos médios de resolução maiores?
6. H0 Certos estados possuem mais reclamações em determinado produtos?
""")
#-----------------------
st.divider()
#-----------------------
#tempo medio de resolução
st.markdown("""
    <h4 style="color: #2aa0d3;">Qual o tempo médio de resolução das reclamações?</h4>
    """, unsafe_allow_html=True)
media = df['Resolution time(in days)'].mean()
st.write(f"🔹 Média do tempo de resolução: **{media:.2f} dias**")
#------------------------
st.divider()
#------------------------
#Estados mais reclamooes
st.markdown("""
    <h4 style="color: #2aa0d3;">Quais os três estados com mais reclamações</h4>
    """, unsafe_allow_html=True)

st.subheader("📌 Top 3 Estados Com Maiores Constetações")
top_produtos = df["state name"].value_counts().head(3)
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

#--------------------------------------------------
st.divider()
#--------------------------------------------------

st.markdown("""
    <h4 style="color: #2aa0d3;">Existe correlação entre o estado e o tempo de reclamação?</h4>
    """, unsafe_allow_html=True)

# Agrupar dados por estado (média de tempo e total de reclamações)
df_estados = df.groupby('state name').agg(
    Tempo_Medio=('Resolution time(in days)', 'mean'),
    Total_Reclamacoes=('ID', 'count')
).reset_index()

# Calcular correlação entre tempo médio e volume de reclamações
correlacao = df_estados[['Tempo_Medio', 'Total_Reclamacoes']].corr()

# Plot com Plotly (heatmap interativo)
fig = px.imshow(
    correlacao,
    text_auto=True,
    color_continuous_scale='Blues',
    labels=dict(x="Variável", y="Variável", color="Correlação"),
    x=['Tempo Médio', 'Total Recl.'],
    y=['Tempo Médio', 'Total Recl.']
)
fig.update_layout(width=500, height=500)
st.plotly_chart(fig, use_container_width=True)

st.write(f"""🔹   A análise revelou uma correlação insignificante (< 0.1) entre volume de reclamações estaduais e tempo médio de resolução. Isso pode sugerir que as políticas de atendimento parecem ser homogêneas entre estados """)