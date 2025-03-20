#Bibliotecas Necessárias
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer
from sklearn.feature_selection import f_classif, SelectKBest
from numpy import unique
import matplotlib.pyplot as plt
from seaborn import set_theme
from seaborn import despine




#Configuração de título e layout
st.set_page_config(page_title="Aplicação para Previsão de Acidentes terrestres", layout="wide", page_icon="🔎")

st.subheader("Página de Métricas e Explicabilidade do Modelo")


#Funnção Principal para tratamento dos dados e criação do modelo
@st.cache_resource
def load_and_process_data(atributos, binario="MultiClasse"): #Tratamento da classe e remoção de colunas altamente relacionadas
    data = pd.read_csv("insurance.csv").drop(["Unnamed: 0", "ThisCarDam", "SeniorTrain", "Airbag", 
                                                            "Cushioning", "MakeModel", "DrivQuality", "DrivHist", "OtherCarCost", "PropCost"], axis=1)
    
    
    #Primeiros Tratamentos nos dados
    for col in data.columns:                
        if data[col].dtype== bool:
            data.loc[data[col]== True, col] = "Yes"
            data.loc[data[col]== False, col] = "No"

    #Definição Modelo MultiClasse ou Binário            
    if binario=="MultiClasse":                      #Essas são as Classes para o Modelo MultiClasse
        data.loc[data["Accident"].isin(["Mild", "Moderate"]), "Accident"] = "Leve-Moderado"
        data.loc[data["Accident"].isin(["None", None]), "Accident"] = "Sem Ocorrências"
        data["Accident"].fillna("Sem Ocorrências", inplace=True)         #A Coluna "Accident" tem o valor 'None' para representar a não ocorrência de acidentes, precisamos nos certificar de preenchê-los
        data.loc[data["Accident"]== "Severe", "Accident"] = "Severo"    #Labels nesse ponto: (Leve_Moderado, Sem Ocorrências, Severo)
    
    else:       #Classes para o modelo binário
        data.loc[data["Accident"].isin(["Mild", "Moderate", "Severe"]), "Accident"] = "Acidente"
        data.loc[data["Accident"].isin(["None", None]), "Accident"] = "Sem Ocorrências"
        data["Accident"] = data["Accident"].fillna("Sem ocorrências")        


       
    #Divisão e codificação dos Dados         
    X = data.drop("Accident", axis=1)
    y = data["Accident"]
    X_treino, X_teste, y_treino, y_teste = train_test_split(X,y,test_size=0.2, random_state=3215)
    encoders = dict()
    for col in X.columns:
        encoder = LabelEncoder()
        if X_treino[col].dtype=="object":
            X_treino[col] = encoder.fit_transform(X_treino[col])
            X_teste[col] = encoder.transform(X_teste[col])
            encoders[col] = encoder
    
    
    #Seleção de Atributos
    seletor = SelectKBest(f_classif, k=atributos)
    X_treino = seletor.fit(X_treino,y_treino).transform(X_treino)
    X_teste = seletor.transform(X_teste)
    colunas_selecionadas = X.columns[seletor.get_support()]
    

    #Criação do Modelo e geração das Métricas
    
    #Hiper-parâmetros encontrados no Grid Search acrescentando apenas o balanceamento de classe 
    modelo = RandomForestClassifier(bootstrap= False, class_weight= "balanced", criterion= 'gini', max_depth= 10, max_leaf_nodes= None, min_samples_leaf= 1, min_samples_split= 5).fit(X_treino, y_treino)
    previsoes = modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, previsoes)
    classification = classification_report(y_teste, previsoes)
    confusao_calculos = confusion_matrix(y_teste, previsoes)
    

    #Matriz de Confusão
    if binario=="MultiClasse":
        confusao = ConfusionMatrixDisplay.from_estimator(modelo, X_teste, y_teste, xticks_rotation=20, cmap="cividis")                  
        for i in range(confusao_calculos.shape[0]):  #Adição de círculos ao redor da matriz de confusão para destacar os erros e acertos do modelo
            for j in range(confusao_calculos.shape[1]):
                if i != j:
                    plt.gca().add_patch(plt.Circle((j, i), 0.2, color="red", fill=False, linewidth=1))
                else:
                    plt.gca().add_patch(plt.Circle((j, i), 0.2, color="green", fill=False, linewidth=1))
        plt.legend(["Acertos do Modelo", "Erros do Modelo"], bbox_to_anchor=(0.0,0))
        plt.grid([])


    else:
        confusao = ConfusionMatrixDisplay.from_estimator(modelo, X_teste, y_teste, xticks_rotation=20, display_labels=["Ocorrência", "Sem Ocorrências"], cmap="cividis")                
        plt.grid([])
        for i in range(confusao_calculos.shape[0]): #Adição de círculos ao redor da matriz de confusão para destacar os erros e acertos do modelo
            for j in range(confusao_calculos.shape[1]):
                if i != j:  # Apenas valores fora da diagonal
                    plt.gca().add_patch(plt.Circle((j, i), 0.15, color="red", fill=False, linewidth=1))
                else:
                    plt.gca().add_patch(plt.Circle((j, i), 0.15, color="green", fill=False, linewidth=1))            
        plt.legend(["Acertos do Modelo", "Erros do Modelo"], bbox_to_anchor=(0.15, 0.0))

    X_teste = pd.DataFrame(X_teste, columns=colunas_selecionadas) #Dataframe para uso nos gráficos de explicabilidade

        
        
    return encoders, X_teste, y_teste, acuracia, confusao, confusao_calculos, modelo, classification
    


#Criação da Interface do Streamlit
with st.sidebar:
    st.markdown(":green[Configurações Avançadas do Modelo]", help="Clique aqui para acessar \
                \n configurações avançadas do Modelo")
    with st.expander("Clique aqui para acessar"):
        st.markdown("**Dica:** *Esses atributos serão atualizados automaticamente no gráfico de explicabilidade*")
        atributos = st.slider("Selecione o número de Atributos para utilizar (Opcional)", min_value=5, max_value=15, value=10, step=1)
        binario = st.radio("Selecione o Tipo de Modelo: ", options=["MultiClasse", "Binário"], index=1)
    encoders, X_teste, y_teste, acuracia, confusao, confusao_calculos, modelo, classification = load_and_process_data(atributos=atributos, binario=binario)
    st.markdown(":green[Configurações da Explicabilidade]")
    with st.expander("Clique aqui para acessar"):
        st.markdown("<hr style='border:1px solid green'</hr> ", unsafe_allow_html=True)
        st.markdown("**Configurações da Explicabilidade**")
        index_instancia = st.number_input("Digite o número do Cliente que deseja Analisar", min_value=0)
        features = st.number_input("Decida quantas características deseja visualizar", min_value=1, value=5, max_value=10)    
    processar = st.button("Carregar Modelo e Visualizações ")


if processar:
            
    with st.spinner("Aguarde um instante... Carregando o Modelo"):
        
        tab1, tab2 = st.tabs(["Métricas", "Explicabilidade"])
        
        #Primeira Tabulação
        with tab1:
            col1, col2 = st.columns([0.45,0.55], gap="medium")
            with col1:                            
                st.pyplot(confusao.figure_)             
                st.markdown("<hr style='border:1px solid green'> ", unsafe_allow_html=True)
                st.markdown("*A Matriz de Confusão acima exibe na diagonal principal os acertos do modelo. \
                            Os acertos estão destacados em verde enquanto os erros \
                            foram marcados em vermelho para uma distinção mais rápida.*")
                                

            with col2:                
                st.markdown(":green[*Desempenho Geral do Modelo Binário:*]  **Acerta aproximadamente :green[9] de 10 previsões**")
                st.markdown(":green[*Desempenho Geral do Modelo MultiClasse:*]  **Acerta aproximadamente :green[8] de 10 previsões**")
                st.markdown("A página de explicabilidade ao lado pode revelar padrões escondidos pois através dela é possível entender como o modelo toma suas decisões! \
                        Pode ser interessante verificar não só onde o modelo acerta, mas também onde ele está errando e porque! \
                        Entendendo isso seria possível não só entender melhor o comportamento dos clientes, \
                        mas também desenvolver modelos mais acertivos no futuro")
                st.markdown(f":green[Acurácia Aproximada do Modelo:] **{acuracia*100:.2f}** %")                                
                st.markdown("<hr style='border:1px green'hr> ", unsafe_allow_html=True)
                st.markdown("*As métricas abaixo nos mostra uma visão mais abrangente do desempenho do modelo em diferentes cenários*!")
                st.text(f"->{classification}")
                st.markdown(":green[**Precision:**] *De todos classificados em determinada classe quantos de fato pertenciam à ela*")
                st.markdown(":green[**Recall:**] *Capacidade do Modelo de identificar corretamente determinada classe*")
                st.markdown(":green[**F1-Score:**] *Combina harmonicamente Precisão e Recall* ")
                st.markdown(":green[**Support:**] *Número Total de Clientes no conjunto de teste* ")

               
               
        #Segunda Tabulação
        with tab2:
            col1, col2 = st.columns(2, gap="medium") 
        
        #Explicabilidade Local do Modelo
            with col1:
                explainer = LimeTabularExplainer(X_teste.values, feature_names=X_teste.columns, class_names=unique(y_teste), discretize_continuous=True)
                instancia = X_teste.iloc[index_instancia,:]
                exp = explainer.explain_instance(instancia, modelo.predict_proba, num_features=features)
                
                fig = exp.as_pyplot_figure()                   
                set_theme("paper", "white")  
                despine(top=True, right=True, left=False, bottom=False)                                           
                plt.axvline(0, color="black", linewidth=1.0)
                plt.grid(axis="x", linestyle="--", linewidth=0.25, color="grey")
                plt.title(f"Impacto das características na Previsão da Instância: {index_instancia}", fontsize=15, fontweight="bold")
                plt.ylabel("Valor das Características", loc="center", fontsize=10, fontweight="bold")
                plt.xlabel("Impacto na Previsão", loc="center", fontsize=10, fontweight="bold")
                plt.yticks(rotation=30)            
                plt.tight_layout()
                st.markdown("Descubra como cada característica impulsiona a previsão do modelo. Barras verdes revelam um efeito positivo, \
                            enquanto barras vermelhas destacam influências negativas")
                st.pyplot(fig, use_container_width=True)          
                

                #Transformações nos dados 
                #Serão necessárias várias transformações para permitir uma visualização mais abrangente dos dados
                instancia_df = pd.DataFrame([instancia], columns=X_teste.columns)
                instancia_dec = instancia_df.copy()
                for col in instancia_dec.columns:
                    if col in encoders:
                        instancia_dec[col] = encoders[col].inverse_transform([instancia_dec[col]])[0]
                
                #Exibição dos Valores Originais
                st.markdown("<hr style='border:1px solid green'> ", unsafe_allow_html=True)
                st.markdown(":green[**Descrição:**] Valores Originais do Cliente para verificação")
                st.table(instancia_dec)
                
                #Previsão e Visualização dos dados
                previsao = modelo.predict(instancia_df.values.reshape(1,-1))            
                classe_original = pd.DataFrame(y_teste).iloc[index_instancia]                
                st.markdown(f"*Ocorrência Real*:  **{classe_original.iloc[0]}**  ==========  *Previsão do Modelo:*  **{previsao[0]}** 🚀 ", unsafe_allow_html=True)
                if previsao[0]!= classe_original.values:
                    st.error("A Previsão do Modelo é diferente da Ocorrência Real!", icon="🚨")
                    st.markdown(" :red[**Dica:**] *Compare os valores e contribuições das características para identificar discrepâncias!*" )
                else:
                    st.success("A Previsão do Modelo está correta!", icon="✅")
                    st.markdown(":green[**Dica:**] *Veja como os Valores dos atributos acima contribuiram para a decisão do Modelo*")

                    

            
            #Explicabilidade Local
            with col2:
                importancia = modelo.feature_importances_                
                importancia = pd.DataFrame({"caracteristicas": X_teste.columns, "importancia": importancia}).nlargest(features, "importancia").sort_values("importancia")
                fig_import = plt.figure()                
                set_theme("paper", "white" )                     
                plt.barh(importancia["caracteristicas"], importancia["importancia"], align="center", color="green", height=0.7)
                despine(top=True, right=True, left=False, bottom=False)                   
                plt.grid(axis="x", linestyle=":", linewidth=0.35, color="grey")
                plt.title("Impacto Geral das Características", fontsize=15, fontweight="bold") 
                plt.xlabel("Impacto Geral nas escolhas do Modelo", fontsize=10, fontweight="bold")
                plt.ylabel("Nome das Características", fontsize=10, fontweight="bold")                
                plt.tight_layout()
                plt.xticks(rotation=0)            
                st.markdown("Entenda a força de cada característica no panorama geral. Este gráfico destaca as principais influências que impactam nas decisões do modelo.")
                st.pyplot(fig_import, clear_figure=False, use_container_width=True)
                st.markdown("<hr style='border:1px solid green'> ", unsafe_allow_html=True)
                st.markdown(":green[**Descrição:**]  *O gráfico acima mostra a importância das características para as previsões do modelo. \
                            Como podemos ver no gráfico acima atributos como as habilidades do motorista e o custo do carro são \
                            características mandatórias para a ocorrência ou não de Acidentes. Então, talvez seja uma boa tática investir em \
                            campanhas para incentivar a melhor qualificação dos motoristas, bem como incentivar clientes sobre a aquisição de carros \
                             com mais dispositivos de segurança que naturalmente tendem a ser mais caros*")
                        
                    
                    
        #Variáveis que precisam ser compartilhadas entre as páginas da aplicação
        st.session_state["X_teste"] = X_teste
        st.session_state["encoders"] = encoders
        st.session_state["modelo"] = modelo
        st.session_state["confusao_calculos"] = confusao_calculos
            

            



        
        




