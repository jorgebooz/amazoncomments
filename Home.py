import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px


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
        "Identificador", "Qualitativa", "Qualitativa", "Qualitativa", "Qualitativa", "Qualitativa",
        "Data(Qualitativa)", "Data (Qualitativa)", "Tempo (Qualitativa)", "Qualitativa",
        "Categorica", "Quantitativa", "Data", "Categorica"
    ]
}

df_tipos = pd.DataFrame(dados_tipos)

# Exibindo no Streamlit
st.subheader("Classificação das Variáveis")
st.dataframe(df_tipos)


st.markdown("<h1 style='color: #2aa0d3;'>Questionamentos e Hipóteses</h1>", unsafe_allow_html=True)

st.write("""
1. Qual o tempo médio de resolução das reclamações?
2. Quais os três produtos mais reclamados?
3. Existe correlação entre o tempo de resposta e a contestação pelo consumidor?
4. Como os tempos de resolução estão distribuídos? Eles seguem uma distribuição normal?
5. Reclamações feitas em certos estados têm tempos médios de resolução maiores?
6. Consumidores de certos produtos tendem a contestar mais a resposta da empresa?
""")

#-----------------------
st.markdown("<h1 style='color: #2aa0d3;'>📊 Intervalo de Confiança do Tempo de Resolução</h1>", unsafe_allow_html=True)


df['Resolution time(in days)'] = pd.to_numeric(df['Resolution time(in days)'], errors='coerce')
res_time = df['Resolution time(in days)'].dropna()
df['Resolution time(in days)'] = pd.to_numeric(df['Resolution time(in days)'], errors='coerce')
df['Resolution time(in days)'] = df['Resolution time(in days)'].apply(lambda x: max(x, 0) if pd.notnull(x) else x)

res_time = df['Resolution time(in days)'].dropna()

media = res_time.mean()
std = res_time.std(ddof=1)
n = len(res_time)

conf = st.slider("Selecione o nível de confiança (%)", 80, 99, 95)
alpha = 1 - (conf / 100)
z = stats.norm.ppf(1 - alpha/2)
margem_erro = z * (std / np.sqrt(n))
lim_inf = media - margem_erro
lim_sup = media + margem_erro

st.write(f"🔹 Média do tempo de resolução: **{media:.2f} dias**")
st.write(f"🔹 Intervalo de Confiança ({conf}%): de **{lim_inf:.2f}** até **{lim_sup:.2f}** dias")
st.write(f"🔹 Margem de erro: ± **{margem_erro:.2f} dias**")

fig = go.Figure()

fig.add_trace(go.Histogram(
    x=res_time,
    nbinsx=40,
    name='Tempo de Resolução',
    marker_color='lightblue',
    opacity=0.75
))

fig.add_vline(x=lim_inf, line_dash="dash", line_color="red", annotation_text="Limite Inferior", annotation_position="top left")
fig.add_vline(x=lim_sup, line_dash="dash", line_color="green", annotation_text="Limite Superior", annotation_position="top right")
fig.add_vline(x=media, line_color="black", annotation_text="Média", annotation_position="top")

fig.update_layout(
    title="Distribuição do Tempo de Resolução com Intervalo de Confiança",
    xaxis_title="Dias",
    yaxis_title="Frequência",
    bargap=0.1
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
    <h4 style="color: #2aa0d3;">Qual o tempo médio de resolução das reclamações?</h4>
    <p>O tempo médio de resolução é de 2 dias, sendo 95% dos casos resolvidos entre 1.9 e 2.1 dias</p>
    """, unsafe_allow_html=True)

