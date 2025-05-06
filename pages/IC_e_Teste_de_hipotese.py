import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

st.image("logo.svg", width=150)

df = pd.read_csv("dataset_finance.csv")

df.columns = df.columns.str.strip()


#IC
st.markdown("<h1 style='color: #2aa0d3;'>📊 Intervalo de Confiança do Tempo de Resolução</h1>", unsafe_allow_html=True)

# Limpeza e preparação dos dados
df['Resolution time(in days)'] = pd.to_numeric(df['Resolution time(in days)'], errors='coerce')
df['Resolution time(in days)'] = df['Resolution time(in days)'].apply(lambda x: max(x, 0) if pd.notnull(x) else x)
res_time = df['Resolution time(in days)'].dropna()

# Cálculos básicos
media = res_time.mean()
std = res_time.std(ddof=1)
n = len(res_time)

# Teste de normalidade
_, p_valor = shapiro(res_time.sample(min(5000, len(res_time))))  # Limita a 5k pontos por performance
normal = p_valor > 0.05

# Widget para seleção de confiança
conf = st.slider("Selecione o nível de confiança (%)", 80, 99, 95)
alpha = 1 - (conf / 100)

# Container para métricas
metric_col1, metric_col2 = st.columns(2)

with metric_col1:
    st.metric("Média (dias)", f"{media:.2f}")
    
with metric_col2:
    st.metric("Tamanho da Amostra", n)
    st.metric("Normalidade (Shapiro-Wilk)", "Normal" if normal else "Não-Normal", 
              f"p-valor = {p_valor:.4f}")

# Seção de análise principal
if normal or n > 30:  # Se normal ou amostra grande (TCL aplicável)
    if normal:
        st.success("✅ Os dados parecem normais (p-valor > 0.05). Podemos usar métodos paramétricos.")
    else:
        st.warning("⚠️ Dados não-normais, mas como n > 30, o Teorema Central do Limite permite usar aproximação normal.")
    
    # Cálculo paramétrico
    if n <= 30:
        t = stats.t.ppf(1 - alpha/2, df=n-1)
        margem_erro = t * (std / np.sqrt(n))
    else:
        z = stats.norm.ppf(1 - alpha/2)
        margem_erro = z * (std / np.sqrt(n))
    
    lim_inf = media - margem_erro
    lim_sup = media + margem_erro
    
    st.subheader("Intervalo de Confiança Paramétrico")
    st.write(f"🔹 Método: {'Distribuição t-Student' if n <= 30 else 'Distribuição Normal (Z)'}")
    st.write(f"🔹 IC ({conf}%): [{lim_inf:.2f}, {lim_sup:.2f}] dias")
    st.write(f"🔹 Margem de erro: ±{margem_erro:.2f} dias")
    
else:
    st.error("🚨 Dados não-normais com amostra pequena (n ≤ 30). Métodos paramétricos podem ser inadequados.")
    
    # Cálculo não-paramétrico (Bootstrap)
    st.subheader("Intervalo de Confiança Não-Paramétrico (Bootstrap)")
    
    def mean_statistic(data):
        return np.mean(data)
    
    bootstrap_ci = bootstrap(
        (res_time,), 
        statistic=mean_statistic, 
        confidence_level=conf/100,
        method='percentile',
        n_resamples=5000
    )
    
    st.write(f"🔹 Método: Bootstrap percentílico (5,000 reamostras)")
    st.write(f"🔹 IC ({conf}%): [{bootstrap_ci.confidence_interval.low:.2f}, {bootstrap_ci.confidence_interval.high:.2f}] dias")
    
    # Também mostramos IC baseado na
    def median_statistic(data):
        return np.median(data)
    
    bootstrap_ci_median = bootstrap(
        (res_time,), 
        statistic=median_statistic, 
        confidence_level=conf/100,
        method='percentile',
        n_resamples=5000
    )

# Visualizações
st.subheader("Visualização da Distribuição")

tab1, tab2, tab3 = st.tabs(["Histograma", "Boxplot", "Violin Plot"])

with tab1:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=res_time,
        nbinsx=40,
        marker_color='lightblue',
        opacity=0.75
    ))
    
    if normal or n > 30:
        fig_hist.add_vline(x=lim_inf, line_dash="dash", line_color="red")
        fig_hist.add_vline(x=lim_sup, line_dash="dash", line_color="red")
    
    fig_hist.update_layout(
        title="Distribuição do Tempo de Resolução",
        xaxis_title="Dias",
        yaxis_title="Frequência"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    fig_box = px.box(res_time, points="all", 
                    title="Boxplot - Distribuição e Outliers")
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    fig_violin = px.violin(res_time, box=True, points="all",
                          title="Distribuição de Densidade")
    fig_violin.update_layout(showlegend=False)
    st.plotly_chart(fig_violin, use_container_width=True)

# Interpretação
if normal or n > 30:
    st.write(f"""
    Com {conf}% de confiança, o tempo médio real de resolução está entre **{lim_inf:.2f}** e **{lim_sup:.2f}** dias.
    - Valores acima de **{lim_sup:.2f}** dias podem indicar atrasos críticos.
    """)
else:
    st.write(f"""
    Para estes dados não-normais (p-valor = {p_valor:.4f}):
    - O intervalo bootstrap para a média é **[{bootstrap_ci.confidence_interval.low:.2f}, {bootstrap_ci.confidence_interval.high:.2f}]** dias.
    - A mediana de **{mediana:.2f}** dias (IC: [{bootstrap_ci_median.confidence_interval.low:.2f}, {bootstrap_ci_median.confidence_interval.high:.2f}]) é uma métrica mais robusta.
    - Considere investigar os outliers, que podem estar distorcendo a média.
    """)

st.write("""
**Recomendações:**
1. Para acompanhamento operacional, use a **mediana** como métrica principal.
2. Monitore casos acima do limite superior como potenciais problemas críticos.
3. Para amostras pequenas não-normais, prefira métodos não-paramétricos.
""")

#teste de hipotese
import pandas as pd
import scipy.stats as stats
import plotly.express as px
import streamlit as st

# Supondo que df seja seu DataFrame principal
# Caso não seja, substitua pelo nome correto do seu DataFrame

# 1. Primeiro criamos o state_stats (que estava faltando)
state_stats = df.groupby('State').agg(
    total_reclamacoes=('ID', 'count'),
    tempo_medio_resolucao=('Resolution time(in days)', 'mean')
).reset_index()

# --------------------------
# ANÁLISE 1: CORRELAÇÃO ENTRE VOLUME E TEMPO DE RESOLUÇÃO
# --------------------------

st.header("1. Análise de Correlação entre Volume e Tempo de Resolução")

# Hipóteses
st.subheader("Hipóteses:")
st.write("- **H₀ (Nula):** Não há correlação entre o volume de reclamações e o tempo médio de resolução nos estados")
st.write("- **H₁ (Alternativa):** Existe correlação positiva entre volume de reclamações e tempo médio de resolução")

# Teste de Spearman (não-paramétrico)
corr, p_valor = stats.spearmanr(
    state_stats['total_reclamacoes'], 
    state_stats['tempo_medio_resolucao']
)

# Resultados
st.subheader("Resultado do Teste de Spearman:")
st.write(f"**Coeficiente de Correlação:** {corr:.3f}")
st.write(f"**p-valor:** {p_valor:.4f}")

# Interpretação
alpha = 0.05
if p_valor < alpha:
    st.success("✅ **Rejeitamos H₀**: Há evidências de correlação significativa (p = {:.4f})".format(p_valor))
    st.write("**Direção da correlação:**", "Positiva" if corr > 0 else "Negativa")
else:
    st.error("❌ **Não rejeitamos H₀**: Sem evidências de correlação (p = {:.4f})".format(p_valor))

# Visualização
fig1 = px.scatter(state_stats, x='total_reclamacoes', y='tempo_medio_resolucao',
                 trendline="lowess", title="Correlação: Volume vs Tempo de Resolução",
                 labels={'total_reclamacoes': 'Número de Reclamações',
                        'tempo_medio_resolucao': 'Tempo Médio (dias)'},
                 hover_name='State')
st.plotly_chart(fig1, use_container_width=True)

# --------------------------
# ANÁLISE 2: TESTE T ENTRE ESTADOS COM MAIS/MENOS RECLAMAÇÕES
# --------------------------

st.header("2. Comparação de Tempos de Resolução entre Grupos de Estados")

# Dividir estados em grupos (alto/baixo volume)
median_reclamacoes = state_stats['total_reclamacoes'].median()
alto_volume = state_stats[state_stats['total_reclamacoes'] > median_reclamacoes]
baixo_volume = state_stats[state_stats['total_reclamacoes'] <= median_reclamacoes]

# Hipóteses
st.subheader("Hipóteses:")
st.write("- **H₀ (Nula):** Não há diferença nos tempos médios de resolução entre estados com alto e baixo volume de reclamações")
st.write("- **H₁ (Alternativa):** Estados com alto volume de reclamações têm tempos médios de resolução diferentes")

# Teste T para amostras independentes
t_stat, p_valor_t = stats.ttest_ind(
    alto_volume['tempo_medio_resolucao'],
    baixo_volume['tempo_medio_resolucao'],
    equal_var=False  # Teste de Welch (não assume variâncias iguais)
)

# Resultados
st.subheader("Resultado do Teste T:")
st.write(f"**Estatística T:** {t_stat:.3f}")
st.write(f"**p-valor:** {p_valor_t:.4f}")

# Interpretação
if p_valor_t < alpha:
    st.success(f"✅ **Rejeitamos H₀**: Diferença significativa (p = {p_valor_t:.4f})")
    # Calculando a direção da diferença
    media_alto = alto_volume['tempo_medio_resolucao'].mean()
    media_baixo = baixo_volume['tempo_medio_resolucao'].mean()
    st.write(f"**Estados com alto volume** têm tempo médio {'maior' if media_alto > media_baixo else 'menor'} ({media_alto:.1f} vs {media_baixo:.1f} dias)")
else:
    st.error(f"❌ **Não rejeitamos H₀**: Sem diferença significativa (p = {p_valor_t:.4f})")

# Visualização
fig2 = px.box(
    pd.concat([
        alto_volume.assign(Grupo='Alto Volume'),
        baixo_volume.assign(Grupo='Baixo Volume')
    ]),
    x='Grupo',
    y='tempo_medio_resolucao',
    title="Distribuição de Tempos de Resolução por Grupo",
    labels={'tempo_medio_resolucao': 'Tempo Médio (dias)'}
)
st.plotly_chart(fig2, use_container_width=True)