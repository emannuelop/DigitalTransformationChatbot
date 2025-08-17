import streamlit as st

st.title("Chatbot de Transformação Digital - TCC")
name = st.text_input("Digite seu nome:")
if name:
    st.write(f"Olá, {name}! Bem-vindo ao protótipo do chatbot 🚀")
