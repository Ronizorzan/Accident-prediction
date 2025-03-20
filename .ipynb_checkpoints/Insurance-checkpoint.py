import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder
from st_aggrid import AgGrid
from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components
from numpy import unique
import matplotlib.pyplot as plt




st.set_page_config(page_title="Classificação de Acidentes terrestres", layout="wide")


@st.cache_resource
def load_and_process_data(binario=False):
    data = pd.read_csv("insurance.csv", encoding="utf-8").drop(["Unnamed: 0", "ThisCarDam", "SeniorTrain", "Airbag", ], axis=1)
                                                               # "Cushioning", "MakeModel", "DrivQuality", "DrivHist", "OtherCarCost", "PropCost", "ThisCarCost"], axis=1)
    data.loc[data["Accident"].isin(["None", None]), "Accident"] = "No Accident"
    for col in data.columns:
        if data[col].dtype== bool:
            data.loc[data[col]== True, col] = "Yes"
            data.loc[data[col]== False, col] = "No"
    if binario==False:
        data = data
    else:
        data.loc[data["Accident"]!= "No Accident", "Accident"] = "Accidented"
    X = data.drop("Accident", axis=1)
    y = data["Accident"]
    X_treino, X_teste, y_treino, y_teste = train_test_split(X,y,test_size=0.2)
    encoders = dict()
    for col in X.columns:
        encoder = LabelEncoder()
        if X_treino[col].dtype=="object":
            X_treino[col] = encoder.fit_transform(X_treino[col])
            X_teste[col] = encoder.transform(X_teste[col])
            encoders[col] = encoder
    y_treino = y_treino.astype("category").cat.codes
    y_teste = y_teste.astype("category").cat.codes
    classes = list(data["Accident"].astype("category").cat.categories)
    modelo = RandomForestClassifier(class_weight="balanced").fit(X_treino, y_treino)
    previsoes = modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, previsoes)
    if binario==False:
        confusao = ConfusionMatrixDisplay.from_estimator(modelo, X_teste, y_teste, display_labels=["Mild", "Moderate", "No Accident", "Severe"], xticks_rotation=45)
    else:
        confusao = ConfusionMatrixDisplay.from_estimator(modelo, X_teste, y_teste, display_labels=["Accidented", "No Accident"], xticks_rotation=45)

        

    return encoders, X_teste, y_teste, acuracia, confusao, modelo, classes
    

with st.sidebar.expander("Configurações do Modelo"):
    binario = st.checkbox("Transformar em classificação binária'", value=False)
    encoders, X_teste, y_teste, acuracia, confusao, modelo, classes = load_and_process_data(binario=binario)
    index_instancia = st.number_input("Escolha uma instância para explicar", min_value=0)
    features = st.number_input("Selecione o número de características para exibir", min_value=3)
    processar = st.button("Processar")
if processar:
    tab1, tab2 = st.tabs(["Modelo", "Explicabilidade"])
    with tab1:
        col1, col2 = st.columns([0.6,0.4])
        with col1:
            st.pyplot(confusao.figure_)

        with col2:
            st.write(f"Acurácia do Modelo: {acuracia:.2f}")
            

    with tab2:
        #explainer = LimeTabularExplainer(X_teste.values, feature_names=X_teste.columns, discretize_continuous=True)
        #previsor = lambda x: modelo.predict_proba(x).astype(float)
        #expl = explainer.explain_instance(X_teste.iloc[0,:], previsor, top_labels=4)        
        #components.html(expl.as_html(), width=600, height=600, scrolling=True)
        col1, col2 = st.columns([0.7,0.3]) 
        with col1:
            explainer = LimeTabularExplainer(X_teste.values, feature_names=X_teste.columns, class_names=unique(classes), discretize_continuous=True)
            instancia = X_teste.iloc[index_instancia,:]
            exp = explainer.explain_instance(instancia, modelo.predict_proba, num_features=features)
            st.write("Explicação para a amostra:")
            #st.write(exp.as_dict())  # Exemplo para a primeira classe
            #st.pyplot(exp.as_pyplot_figure())
            fig = exp.aspyplot_figure()
            plt.title(f"Explicação do Modelo para a instância: {index_instancia}")
            plt.tight_layout()
            colors = ["green" if color > 0 else "red" for color in exp.as_list()]
            st.pyplot(fig, use_container_width=True)
            #components.html(exp.as_html(), height=800, width=800)
            instancia_df = pd.DataFrame([instancia], columns=X_teste.columns)
            instancia_dec = instancia_df.copy()
            for col in instancia_dec.columns:
                if col in encoders:
                    instancia_dec[col] = encoders[col].inverse_transform([instancia_dec[col]])[0]
            previsao = modelo.predict(instancia_df.values.reshape(1,-1))
            st.write(y_teste.astype("category").cat.categories[previsao][0])
            classe_original = y_teste.iloc[index_instancia]
            if previsao[0]!= classe_original:
                st.error("A Previsão do Modelo é diferente da Original... Verifique os valores abaixo para verificar o erro do modelo")
            else:
                st.success("A Previsão do Modelo está correta... Verifique os dados Originais abaixo e tire suas conclusões")
            st.markdown("**Descrição:** Linha Escolhida com os valores originais")
            st.write(instancia_dec)


        
        




