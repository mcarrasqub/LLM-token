import time
import streamlit as st
import tiktoken
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from groq import Groq
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Desmontando los LLMs",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #313244;
    }
    .token-display {
        font-family: monospace;
        font-size: 1rem;
        line-height: 2.4;
        padding: 12px;
        background: #1e1e2e;
        border-radius: 8px;
    }
    .section-intro {
        color: #a6adc8;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Desmontando los LLMs")
    st.caption("Universidad EAFIT · 2026-1")
    st.divider()

    st.subheader("🔑 Credenciales")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Obtén tu clave gratuita en console.groq.com",
    )

    st.divider()
    st.subheader("🤖 Modelo")
    model_choice = st.selectbox(
        "Selecciona el modelo",
        options=[
            "llama-3.1-8b-instant",
            "llama3-8b-8192",
            "gemma2-9b-it",
            "mistral-saba-24b",
            "llama-3.3-70b-versatile",
        ],
        index=0,
        help="Modelos de bajo costo / pocos parámetros disponibles en Groq",
    )
    st.caption(f"Usando: `{model_choice}`")

    st.divider()
    st.markdown(
        "📚 [Groq Cloud](https://console.groq.com) · "
        "[Tiktoken](https://github.com/openai/tiktoken) · "
        "[Sentence Transformers](https://sbert.net)"
    )

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.title("🧠 Desmontando los LLMs")
st.markdown(
    "**Taller Técnico · Deep Learning y Arquitecturas Transformer** · "
    "Prof. Jorge Ivan Padilla Buritica · Universidad EAFIT"
)
st.divider()

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔤 Módulo 1 · Tokenizador",
    "📐 Módulo 2 · Embeddings",
    "🤖 Módulo 3 · Inferencia",
    "📊 Módulo 4 · Métricas",
])

# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 1 · TOKENIZADOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🔤 El Laboratorio del Tokenizador")
    st.markdown(
        '<p class="section-intro">'
        "La tokenización convierte texto crudo en IDs numéricos que el modelo puede procesar. "
        "Cada token puede ser una palabra completa, una sílaba, un signo de puntuación o incluso un espacio."
        "</p>",
        unsafe_allow_html=True,
    )

    col_input, col_enc = st.columns([3, 1])
    with col_input:
        user_text = st.text_area(
            "✏️ Ingresa tu texto:",
            value="Los transformers han revolucionado el procesamiento del lenguaje natural desde 2017.",
            height=110,
        )
    with col_enc:
        encoding_name = st.selectbox(
            "Encoding",
            ["cl100k_base", "p50k_base", "r50k_base"],
            help="cl100k_base es el usado por GPT-4 y modelos modernos.",
        )

    if user_text.strip():
        try:
            enc = tiktoken.get_encoding(encoding_name)
            token_ids = enc.encode(user_text)
            token_strings = [enc.decode([t]) for t in token_ids]

            # ── Colored tokens ───────────────────────────────────────────────
            st.subheader("Tokens coloreados")
            COLORS = [
                "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
                "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F", "#82E0AA",
            ]
            html_tokens = '<div class="token-display">'
            for i, (tok_str, tok_id) in enumerate(zip(token_strings, token_ids)):
                color = COLORS[i % len(COLORS)]
                display = tok_str.replace("<", "&lt;").replace(">", "&gt;").replace(" ", "·")
                html_tokens += (
                    f'<span title="ID: {tok_id}" style="'
                    f"background:{color}22;border:1px solid {color};"
                    f"padding:2px 6px;margin:2px;border-radius:5px;"
                    f'display:inline-block;color:{color}">{display}</span>'
                )
            html_tokens += "</div>"
            st.markdown(html_tokens, unsafe_allow_html=True)

            # ── Token → ID table ─────────────────────────────────────────────
            st.subheader("Mapeo Token → ID")
            df_tokens = pd.DataFrame({
                "Posición": range(len(token_ids)),
                "Token (repr.)": [repr(t) for t in token_strings],
                "Token ID": token_ids,
            })
            st.dataframe(df_tokens, use_container_width=True, height=220)

            # ── Metrics ──────────────────────────────────────────────────────
            st.subheader("Métricas comparativas")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📝 Caracteres", len(user_text))
            c2.metric("🔢 Tokens", len(token_ids))
            c3.metric("📊 Ratio chars/token", f"{len(user_text)/len(token_ids):.2f}")
            words = len(user_text.split())
            c4.metric("📖 Palabras", words)

            # ── Bar chart ────────────────────────────────────────────────────
            fig_bar = go.Figure(go.Bar(
                x=["Caracteres", "Palabras", "Tokens"],
                y=[len(user_text), words, len(token_ids)],
                marker_color=["#45B7D1", "#96CEB4", "#FF6B6B"],
                text=[len(user_text), words, len(token_ids)],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="Comparación: Caracteres vs Palabras vs Tokens",
                template="plotly_dark",
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"Error al tokenizar: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 2 · EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📐 Geometría de las Palabras")
    st.markdown(
        '<p class="section-intro">'
        "Los embeddings mapean palabras a vectores en un espacio de alta dimensionalidad donde la "
        "<em>distancia semántica</em> tiene significado algebraico. Usamos PCA para proyectarlos a 2D."
        "</p>",
        unsafe_allow_html=True,
    )

    @st.cache_resource(show_spinner="Cargando modelo de embeddings (primera vez tarda ~30s)...")
    def load_embedding_model():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")

    words_input = st.text_input(
        "📝 Palabras a visualizar (separadas por comas):",
        value="rey, reina, hombre, mujer, príncipe, princesa, niño, niña, abuelo, abuela",
    )

    col_emb_btn, col_emb_info = st.columns([1, 3])
    with col_emb_btn:
        run_emb = st.button("🔍 Generar visualización", use_container_width=True)
    with col_emb_info:
        st.info("ℹ️ El modelo `all-MiniLM-L6-v2` genera vectores de 384 dimensiones por palabra.")

    if run_emb:
        words = [w.strip() for w in words_input.split(",") if w.strip()]
        if len(words) < 3:
            st.warning("Ingresa al menos 3 palabras para una visualización significativa.")
        else:
            with st.spinner("Calculando embeddings..."):
                model_emb = load_embedding_model()
                embeddings = model_emb.encode(words, show_progress_bar=False)

            # PCA 2D
            pca = PCA(n_components=2, random_state=42)
            coords_2d = pca.fit_transform(embeddings)

            df_emb = pd.DataFrame({
                "Palabra": words,
                "PC1": coords_2d[:, 0],
                "PC2": coords_2d[:, 1],
            })

            fig_emb = px.scatter(
                df_emb, x="PC1", y="PC2", text="Palabra",
                title="Espacio de Embeddings proyectado a 2D (PCA)",
                template="plotly_dark",
                color_discrete_sequence=["#4ECDC4"],
            )
            fig_emb.update_traces(
                textposition="top center",
                marker=dict(size=14, line=dict(width=1, color="#ffffff")),
                textfont=dict(size=13),
            )
            fig_emb.update_layout(height=520, showlegend=False)
            st.plotly_chart(fig_emb, use_container_width=True)

            var = pca.explained_variance_ratio_
            st.caption(
                f"Varianza explicada → PC1: **{var[0]:.1%}** · PC2: **{var[1]:.1%}** · "
                f"Total: **{sum(var):.1%}**"
            )

            # ── Analogía rey - hombre + mujer ≈ reina ───────────────────────
            word_lower = [w.lower() for w in words]
            analogy_targets = {
                "es": ["rey", "hombre", "mujer", "reina"],
                "en": ["king", "man", "woman", "queen"],
            }
            found_set = None
            for lang, targets in analogy_targets.items():
                if all(t in word_lower for t in targets):
                    found_set = targets
                    break

            if found_set:
                st.subheader("🔢 Verificación de analogía vectorial")
                a, b, c, d = found_set
                idx = {w: word_lower.index(w) for w in found_set}
                analogy_vec = (
                    embeddings[idx[a]] - embeddings[idx[b]] + embeddings[idx[c]]
                )
                sim = cosine_similarity([analogy_vec], [embeddings[idx[d]]])[0][0]

                col_a, col_b = st.columns([1, 2])
                with col_a:
                    label = f"v({a}) − v({b}) + v({c}) ≈ v({d})"
                    st.metric(f"Similitud coseno", f"{sim:.4f}", help=label)
                    if sim > 0.85:
                        st.success("✅ Relación algebraica confirmada")
                    elif sim > 0.65:
                        st.info("🔶 Relación aproximada detectada")
                    else:
                        st.warning("⚠️ Relación débil en este espacio reducido")
                with col_b:
                    st.latex(
                        r"\vec{v}(\text{" + a + r"}) - \vec{v}(\text{" + b + r"}) + "
                        r"\vec{v}(\text{" + c + r"}) \approx \vec{v}(\text{" + d + r"})"
                    )
            else:
                st.info(
                    "💡 Incluye **rey, hombre, mujer, reina** (o king/man/woman/queen) "
                    "para verificar la analogía vectorial."
                )

            # ── Heatmap de similitud coseno ──────────────────────────────────
            with st.expander("🗺️ Mapa de calor de similitudes entre palabras"):
                sim_matrix = cosine_similarity(embeddings)
                fig_heat = px.imshow(
                    sim_matrix,
                    x=words, y=words,
                    color_continuous_scale="RdBu",
                    zmin=-1, zmax=1,
                    title="Similitud Coseno entre palabras",
                    template="plotly_dark",
                )
                fig_heat.update_layout(height=450)
                st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 3 · INFERENCIA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🤖 Inferencia y Razonamiento")
    st.markdown(
        '<p class="section-intro">'
        "Experimenta cómo los hiperparámetros de generación afectan las respuestas del modelo. "
        "La <strong>temperatura</strong> controla la aleatoriedad; <strong>Top-P</strong> limita el vocabulario activo."
        "</p>",
        unsafe_allow_html=True,
    )

    if not api_key:
        st.warning("⚠️ Ingresa tu **Groq API Key** en el panel lateral para continuar.")
        st.stop()

    col_params, col_chat = st.columns([1, 2], gap="large")

    with col_params:
        st.subheader("⚙️ Parámetros")

        temperature = st.slider(
            "🌡️ Temperatura",
            min_value=0.0, max_value=2.0, value=0.7, step=0.05,
            help="< 0.3 → determinista | > 0.7 → creativo",
        )
        if temperature < 0.3:
            st.info("🎯 **Determinista**: respuestas consistentes y factuales.")
        elif temperature < 0.7:
            st.success("⚖️ **Balanceado**: creatividad moderada.")
        else:
            st.warning("🎨 **Creativo**: alta variabilidad y originalidad.")

        top_p = st.slider(
            "🎯 Top-P (Nucleus Sampling)",
            min_value=0.05, max_value=1.0, value=0.9, step=0.05,
            help="Acumula tokens hasta cubrir P% de probabilidad acumulada.",
        )
        max_tokens_out = st.slider(
            "📏 Máx. tokens de salida",
            min_value=50, max_value=2048, value=512, step=50,
        )

        st.divider()
        st.subheader("📋 System Prompt")
        system_prompt = st.text_area(
            "Instrucción del sistema:",
            value=(
                "Eres un asistente educativo experto en inteligencia artificial y "
                "deep learning. Responde de manera clara, concisa y en español."
            ),
            height=140,
        )
        st.caption(
            "El **System Prompt** define el rol e instrucciones globales del modelo. "
            "Es la base del *Instruction Tuning*."
        )

    with col_chat:
        st.subheader("💬 Conversación")

        # Historial de conversación en session_state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_prompt = st.chat_input("Escribe tu mensaje aquí...")

        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            client = Groq(api_key=api_key)
            api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

            with st.chat_message("assistant"):
                with st.spinner("Generando..."):
                    t0 = time.time()
                    try:
                        response = client.chat.completions.create(
                            model=model_choice,
                            messages=api_messages,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens_out,
                        )
                        t1 = time.time()
                        answer = response.choices[0].message.content
                        st.markdown(answer)

                        # Save to state
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.session_state["last_response"] = response
                        st.session_state["last_elapsed"] = t1 - t0
                        st.session_state["last_params"] = {
                            "temperature": temperature,
                            "top_p": top_p,
                            "model": model_choice,
                            "system_prompt": system_prompt,
                        }

                    except Exception as e:
                        st.error(f"❌ Error al llamar a Groq: {e}")

        if st.session_state.messages:
            if st.button("🗑️ Limpiar conversación"):
                st.session_state.messages = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 4 · MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("📊 Métricas de Desempeño")
    st.markdown(
        '<p class="section-intro">'
        "Groq utiliza hardware especializado (LPU) que permite velocidades de inferencia muy superiores "
        "a las GPUs tradicionales. Aquí puedes observar las métricas en tiempo real de cada solicitud."
        "</p>",
        unsafe_allow_html=True,
    )

    if "last_response" not in st.session_state:
        st.info("ℹ️ Genera al menos una respuesta en el **Módulo 3** para ver las métricas.")
    else:
        resp = st.session_state["last_response"]
        elapsed = st.session_state["last_elapsed"]
        params = st.session_state.get("last_params", {})
        usage = resp.usage

        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # Groq provides timing fields in usage
        completion_time = getattr(usage, "completion_time", None)
        prompt_time = getattr(usage, "prompt_time", None)
        total_time_api = getattr(usage, "total_time", None)

        if completion_time and completion_time > 0:
            tpt_ms = (completion_time * 1000) / max(completion_tokens, 1)
            throughput = completion_tokens / completion_time
            display_time = total_time_api or elapsed
        else:
            tpt_ms = (elapsed * 1000) / max(completion_tokens, 1)
            throughput = completion_tokens / max(elapsed, 0.001)
            display_time = elapsed

        # ── KPI cards ────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("⏱️ Time per Token", f"{tpt_ms:.1f} ms")
        c2.metric("⚡ Throughput", f"{throughput:.0f} tok/s")
        c3.metric("🕐 Tiempo total", f"{display_time:.2f} s")
        c4.metric("🔢 Total tokens", total_tokens)

        st.divider()

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            # Token breakdown
            fig_tok = go.Figure(go.Bar(
                x=["Prompt (entrada)", "Completion (salida)"],
                y=[prompt_tokens, completion_tokens],
                marker_color=["#4ECDC4", "#FF6B6B"],
                text=[prompt_tokens, completion_tokens],
                textposition="outside",
                width=0.5,
            ))
            fig_tok.update_layout(
                title="Tokens de entrada vs. salida",
                yaxis_title="Tokens",
                template="plotly_dark",
                height=320,
                showlegend=False,
            )
            st.plotly_chart(fig_tok, use_container_width=True)

        with col_chart2:
            # Time breakdown (if available)
            if prompt_time and completion_time:
                fig_time = go.Figure(go.Bar(
                    x=["Prompt processing", "Completion"],
                    y=[prompt_time * 1000, completion_time * 1000],
                    marker_color=["#45B7D1", "#96CEB4"],
                    text=[f"{prompt_time*1000:.0f} ms", f"{completion_time*1000:.0f} ms"],
                    textposition="outside",
                    width=0.5,
                ))
                fig_time.update_layout(
                    title="Desglose de tiempo (ms)",
                    yaxis_title="Milisegundos",
                    template="plotly_dark",
                    height=320,
                    showlegend=False,
                )
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.info("Desglose de tiempo no disponible para este modelo.")

        # ── Request metadata ──────────────────────────────────────────────────
        st.subheader("🗂️ Detalles de la solicitud")
        meta = {
            "modelo": resp.model,
            "temperatura": params.get("temperature", "—"),
            "top_p": params.get("top_p", "—"),
            "finish_reason": resp.choices[0].finish_reason,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "throughput_tokens_per_sec": round(throughput, 2),
            "time_per_token_ms": round(tpt_ms, 2),
        }
        st.json(meta)

        # ── Self-Attention note ───────────────────────────────────────────────
        with st.expander("💡 Conexión con Self-Attention"):
            st.markdown("""
El mecanismo de **Self-Attention** permite que cada token "observe" todos los demás tokens 
del contexto al calcular su representación. Esto significa que:

- Con un **System Prompt largo**, el modelo tiene más contexto inicial y el `prompt_tokens` 
  aumenta, incrementando levemente la latencia de prefill.
- Con conversaciones largas, el contexto crece acumulativamente → mayor `prompt_tokens`.
- El **Throughput** medido aquí refleja la velocidad del LPU de Groq durante la generación 
  **autoregresiva** (un token a la vez).

> *Cambia el System Prompt entre corto y largo en el Módulo 3 y observa cómo varía `prompt_tokens`.*
""")
