import streamlit as st

st.set_page_config(page_title="Assistente Acadêmico", page_icon="🎓", layout="centered")

# Espaçamento
st.markdown("<br><br>", unsafe_allow_html=True)

# Título
st.markdown(
    "<h1 style='text-align: center;'>🎓 Assistente Acadêmico</h1>",
    unsafe_allow_html=True
)

# Subtítulo
st.markdown(
    "<p style='text-align: center; color: gray;'>Escolha como deseja utilizar a plataforma</p>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# Layout em colunas para centralizar
col1, col2 = st.columns(2)

with col1:
    if st.button("👨‍🏫 Criar disciplina", use_container_width=True):
        st.session_state.page = "create_kb"
        st.switch_page("pages/create_kb.py")

with col2:
    if st.button("👩‍🎓 Consultar disciplina", use_container_width=True):
        st.session_state.page = "chat"
        st.switch_page("pages/chat.py")