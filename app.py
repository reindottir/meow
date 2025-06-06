import streamlit as st

pages = {
    Анализ и модель analysis_and_model.py,
    Презентация presentation.py
}

st.sidebar.title(Навигация)
selection = st.sidebar.radio(Страницы, list(pages.keys()))

with open(pages[selection]) as f
    exec(f.read(), globals())