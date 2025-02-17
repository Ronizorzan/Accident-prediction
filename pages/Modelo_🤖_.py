#Bibliotecas
import streamlit as st
from pandas import DataFrame
from time import sleep


#Configuração do Layout
st.set_page_config(page_title="Previsão e prevenção de Acidentes", layout="centered", page_icon="🤖")

st.title("Utilização do Modelo para Previsão")

#Verificação do Modelo
if "modelo" not in st.session_state:
    st.error("O Modelo não foi carregado corretamente! Por favor, tente novamente.")
else:
    X_teste = st.session_state["X_teste"]
    encoders = st.session_state["encoders"]
    modelo = st.session_state["modelo"]
    

    X_teste_decod = DataFrame()  # DataFrame para armazenar os dados decodificados

    # Loop para transformação de X_teste nos valores originais
    for col in X_teste.columns:
        if col in encoders:
            X_teste_decod[col] = encoders[col].inverse_transform(X_teste[col])
        else:
            X_teste_decod[col] = X_teste[col]

    
    novos_dados = []  # Lista para armazenar os novos dados de entrada

    # Loop para criação das caixas de seleção do Modelo
    for col in X_teste_decod.columns:
        insercao = st.selectbox(f"Escolha os dados de ({col}) ", X_teste_decod[col].unique(), key=col)
        novos_dados.append(insercao)

    processar = st.button("Obter a Previsão")
    
    if processar:
        
        #Barra de Progresso
        progress = st.progress(0)        
        for percent_complete in range(100):
            sleep(0.001)
            progress.progress(percent_complete + 1)

            #Criação e tratamento da previsão para posterior visualização
        novo_df = DataFrame([novos_dados], columns=X_teste.columns)
        for col in novo_df.columns:
            if col in encoders:
                novo_df[col] = encoders[col].transform(novo_df[col])
        previsao = modelo.predict(novo_df)
        previsao_porcent = modelo.predict_proba(novo_df)
        
        st.markdown(f"*Previsão de Ocorrência para esse cliente:*  **{previsao[0]}**")
        porcent = (previsao_porcent).max()
        if porcent >0.5:
            st.markdown("<hr style='border:1px solid blue'> ", unsafe_allow_html=True)
            st.text(f"Probabilidade: {porcent*100:.2f}%")
                    








