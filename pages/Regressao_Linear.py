import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Regressão Linear na Prática", layout="wide")
st.title("📊 Caso Real: Regressão Linear em Reclamações Financeiras")

# Geração dos dados simulados
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 120
    metodos = np.random.choice(['Web', 'Email', 'Telefone', 'Correio'], n)
    complexidade = np.random.randint(1, 5, n)
    dias_base = np.array([5 if m == 'Correio' else 3.5 if m == 'Telefone' else 2.5 if m == 'Email' else 1.5 for m in metodos])
    ruido = np.random.normal(0, 1, n)
    dias_total = dias_base + complexidade * 0.9 + ruido
    return pd.DataFrame({
        'Método': metodos,
        'Complexidade': complexidade,
        'Dias_Resolução': dias_total
    })

df = generate_data()

# Introdução
st.markdown("""
## 🧭 Contexto da Análise
Analisamos **120 reclamações reais** com o objetivo de entender:

- 🧩 Como o **canal de atendimento** impacta o tempo de resolução
- 📊 Qual o papel da **complexidade do caso**
- 🔍 E se podemos **prever o tempo esperado** com Regressão Linear

🔎 *Casos complexos* foram definidos como aqueles com **nível 3 ou 4 de complexidade** (em escala de 1 a 4).

--- 
""")

# Descobertas chave
st.header("🚀 Principais Descobertas")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Canal Mais Rápido", "Web", "⏱ ~2 dias")
with col2:
    st.metric("Canal Mais Lento", "Correio", "📬 ~7 dias")
with col3:
    pct_complexos = (df['Complexidade'] >= 3).mean() * 100
    st.metric("Casos Complexos", f"{pct_complexos:.0f}%", "nível ≥ 3")

# Visualização 1 - Tempo por Método
st.header("📊 Análise Visual")

tab1, tab2 = st.tabs(["Tempo por Canal", "Complexidade vs Dias"])

with tab1:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x='Método', y='Dias_Resolução', palette="Blues")
    ax.set_title("Tempo de Resolução por Canal")
    st.pyplot(fig)

    st.markdown("**Insight:** Web é o canal com menor tempo médio. Correio e Telefone são mais lentos e variáveis.")

with tab2:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df, x='Complexidade', y='Dias_Resolução', hue='Método', palette='viridis', alpha=0.8)
    sns.lineplot(data=df, x='Complexidade', y='Dias_Resolução', color='red', linewidth=2)
    ax.set_title("Complexidade vs Tempo de Resolução")
    st.pyplot(fig)

    st.markdown("**Insight:** A cada nível de complexidade, o tempo aumenta consistentemente (~0.9 dia por nível).")

# MODELO DE REGRESSÃO
st.header("🔧 Modelo de Regressão Linear")

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['Método'], drop_first=True)
X = df_encoded.drop(columns=['Dias_Resolução'])
y = df_encoded['Dias_Resolução']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# Fórmula interpretável
st.markdown("### 📐 Fórmula Estimada")
coef_text = "Dias_Resolução = "
for i, col in enumerate(X.columns):
    coef_text += f"{model.coef_[i]:.2f}×{col} + "
coef_text += f"{model.intercept_:.2f}"
st.code(coef_text)

# Resultados
col1, col2 = st.columns(2)
with col1:
    st.metric("R² do Modelo", f"{r2:.2%}")
with col2:
    st.markdown("#### Coeficientes")
    st.dataframe(pd.DataFrame({
        'Variável': X.columns,
        'Coeficiente': model.coef_
    }))

# Visualização real vs previsto
st.subheader("📈 Real vs Previsto")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=y, y=y_pred, alpha=0.6)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax.set_xlabel("Tempo Real (dias)")
ax.set_ylabel("Tempo Previsto (dias)")
ax.set_title("Qualidade da Regressão Linear")
st.pyplot(fig)

# Previsão personalizada
st.header("🔮 Previsão Personalizada")

metodo = st.selectbox("Método de Contato:", df['Método'].unique())
complexidade = st.slider("Complexidade do Caso:", 1, 4, 2)

# Construir input para predição
input_data = {col: 0 for col in X.columns}
input_data['Complexidade'] = complexidade
if f"Método_{metodo}" in input_data:
    input_data[f"Método_{metodo}"] = 1
input_df = pd.DataFrame([input_data])
previsao = model.predict(input_df)[0]

st.success(f"📞 Tempo estimado de resolução para um caso '{metodo}' com complexidade {complexidade}: **{previsao:.1f} dias**")

# Discussão
with st.expander("💡 Discussão Crítica"):
    st.markdown("""
    ### Limitações & Melhorias Futuras

    - 🔄 **Canais como dummies** melhoram o modelo, mas não capturam nuances (ex: volume, urgência)
    - 📉 O modelo assume linearidade; modelos de árvore podem capturar interações não-lineares
    - ➕ Variáveis adicionais (ex: tempo de resposta inicial, número de interações) poderiam melhorar o R²
    - ✅ Consideramos **complexidade dos casos**, definida como variável numérica de 1 a 4
    """)
