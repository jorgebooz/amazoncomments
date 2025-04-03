import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Amazon Analisys", layout="wide")

st.markdown(
    """
    <style>
        html, body, [class*="st-"] {
            background-color: white !important;
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

#--------------------------------------------------

st.image("amazon.svg", width=150)

df = pd.read_csv("datatset_consumer_complaints.csv")

df.columns = df.columns.str.strip()

st.title("Análise de Reclamações de Consumidores da Amazon")
st.write("""
Snalisar sentimentos em compras na Amazon para entender e aprimorar a satisfação dos clientes. A base de dados contém informações detalhadas sobre empresas, produtos, tipos de reclamações, estados e tempo de resolução. O foco será avaliar padrões nos sentimentos expressos pelos consumidores e identificar oportunidades de melhoria. A análise ajudará a propor soluções estratégicas para otimizar a experiência do cliente.
""")

## **1. Apresentação dos Dados**
st.markdown("<h1 style='color: #FFB200;'>1️⃣ Apresentação das variáveis</h1>", unsafe_allow_html=True)
st.subheader("Amostra dos Dados")
st.write(df.head())
st.subheader("Tipos de Variáveis")
st.write(df.dtypes)

st.markdown("<h1 style='color: #FFB200;'>Principais Perguntas de Análise</h1>", unsafe_allow_html=True)
st.subheader("")
st.write("""
1. Qual o tempo médio de resolução das reclamações?
2. Quais os produtos mais reclamados?
3. Existe correlação entre o tempo de resposta e a contestação pelo consumidor?
4. Como os tempos de resolução estão distribuídos? Eles seguem uma distribuição normal?
""")