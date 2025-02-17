#Bibliotecas
import streamlit as st
import matplotlib.pyplot as plt



#Configuração de layout
st.set_page_config(page_title="Relatório Financeiro", layout="wide", page_icon="💵")


st.title("Retorno Financeiro do Modelo" )


if "confusao_calculos" not in st.session_state:
    st.error("Os dados não foram carregados corretamente, por favor tente novamente...")

#Configuração da Barra Lateral 
else:
    with st.sidebar.expander("Insira os Valores para calcular o retorno financeiro"):
        confusao = st.session_state["confusao_calculos"]
        pa = st.number_input("Prêmio Alto - R$:", value=2500)
        pb = st.number_input("Prêmio Baixo - R$:", value=1200)
        cs = st.number_input("Custo Médio por Sinistro - R$:", value=15000)
        taxa_cancelamento = st.slider("Taxa de Cancelamento - Clientes Insatisfeitos (%):", value=10.0, min_value=0.0, max_value=100.0) / 100
        ltv = st.number_input("Valor Médio Vitalício do Cliente (LTV) - R$:", value=8000)
        calcular = st.button("Gerar o Relatório")

    if calcular:

        #localiza os valores da matriz de confusão
        vp = confusao[0][0]
        fn = confusao[0][1]
        fp = confusao[1][0]
        vn = confusao[1][1]
        
        #Verdadeiros Positivos
        receita_vp =( vp * pa )# Cálculo Prêmios Pagos Verdadeiros Positivos
        custo_sinistro_vp = vp * cs #Cálculo Sinistros Pagos
        lucro_vp = receita_vp - custo_sinistro_vp # Receita Gerada menos os sinistros pagos

        # Vercadeiros Negativos
        receita_vn = (vn * pb) # Receita Gerada com os prêmios
        lucro_vn = receita_vn  # Sem custos de sinistro

        # Falsos Positivos
        receita_fp = fp * pa # Prêmios altos Pagos 
        clientes_cancelados = int(fp * taxa_cancelamento) #Clientes Insatisfeitos com Prêmios Altos
        perda_cancelamento = clientes_cancelados * ltv #Perda Financeira com clientes Insatisfeitos 
        lucro_fp = receita_fp - perda_cancelamento #Receita Gerada menos as perdas com cancelamentos

        # Falsos Negativos (Maiores Perdas)
        receita_fn = fn * pb #Lucro com falsos negativos
        custo_sinistro_fn = fn * cs # Sinistros Pagos 
        lucro_fn = receita_fn - custo_sinistro_fn #Receita Gerada menos custos com sinistros 

        total_clientes = vp + vn + fp + fn # Total de Clientes

        clientes_cancelados = int(fp * taxa_cancelamento) # Clientes Cancelados
 
        clientes_retidos = total_clientes - clientes_cancelados # Clientes Retidos

        # Total LTV dos Clientes Retidos
        total_ltv_retido = clientes_retidos * ltv

        # Totais
        receita_total = receita_vp + receita_vn + receita_fp + receita_fn #Cálculo da Receita Gerada
        custo_total = custo_sinistro_vp + custo_sinistro_fn + perda_cancelamento #Cálculo Sinistros Pagos
        lucro_total = receita_total - custo_total + total_ltv_retido #Cálculo Final 

        # Exibição dos Resultados
        col1, col2, col3 = st.columns([0.2,0.35, 0.45])
        
        #Exibição da primeira coluna
        with col1:
                        
            if lucro_total > 0:
                
                st.markdown("<h2 style='color: green;'>Resultados Financeiros Totalizados</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='color: red;'>Resultados Financeiros Totalizados</h2>", unsafe_allow_html=True)


            st.markdown("**Retorno Financeiro Líquido:**")
            if lucro_total > 0:
                st.success(f"R$ {lucro_total:,.2f} 🚀", icon="✅")
            else:
                st.error(f"R$ {lucro_total:,.2f}", icon="🚨")

            st.markdown("<hr style='border:1px solid green'> ", unsafe_allow_html=True)

            st.markdown("**Receitas Totais:**")
            st.text(f"R$ {receita_total:,.2f}")

            st.markdown("**Total LTV dos Clientes Retidos:**")
            st.write(f"R$ {total_ltv_retido:,.2f}")

            st.markdown("**Custos Totais:**")
            st.text(f"R$ {custo_total:,.2f}")


        #Segunda Coluna
        with col2:
                st.subheader("Detalhamento por Categoria")

                st.markdown("**Acidentes corretamente previstos (VP):**")
                st.text(f"Receita Gerada: R$ {receita_vp:,.2f}")
                st.text(f"Custo de Sinistros: R$ {custo_sinistro_vp:,.2f}")
                st.text(f"Retorno Bruto da Categoria: R$ {lucro_vp:,.2f}")

                st.markdown("**Não Acidentes corretamente previstos (VN):**")
                st.text(f"Receita Gerada: R$ {receita_vn:,.2f}")
                st.text(f"Lucro: R$ {lucro_vn:,.2f}")

                st.markdown("**Falsos Alertas de Acidente (FP):**")
                st.text(f"Receita Gerada: R$ {receita_fp:,.2f}")
                st.text(f"Perdas por Cancelamento: R$ {perda_cancelamento:,.2f}")
                st.text(f"Lucro Ajustado para a categoria: R$ {lucro_fp:,.2f}")

                st.markdown("**Acidentes não previstos (FN):**")
                st.text(f"Receita Gerada: R$ {receita_fn:,.2f}")
                st.text(f"Custo de Sinistros: R$ {custo_sinistro_fn:,.2f}")
                st.text(f"Retorno Bruto da Categoria: R$ {lucro_fn:,.2f}")

                if len(confusao)> 2:
                    st.error("Por favor selecione o modelo do tipo Binário na página Classificação para visualizar um relatório preciso")

        

#Terceira coluna
        with col3:
             
            
            # Dados para o gráfico
            categorias = ['Acidentes Previstos', 'Não Acidentes Previstos', 'Falsos Alertas', 'Acidentes não Previstos', 'Clientes Retidos']
            valores = [lucro_vp, lucro_vn, lucro_fp, lucro_fn, total_ltv_retido]

            # Definir cores: verde para lucro, vermelho para prejuízo
            cores = ['green' if valor >= 0 else 'red' for valor in valores]

            # Criar o gráfico
            fig, ax = plt.subplots()
            barras = ax.bar(categorias, valores, color=cores)

            # Adicionar valores acima das barras
            for idx, barra in enumerate(barras):
                altura = barra.get_height()
                ax.text(barra.get_x() + barra.get_width() / 2, altura,
                        f'R$ {valores[idx]:,.2f}',
                        ha='center', va='bottom' if altura >= 0 else 'top',
                        color='black', fontsize=9)

            # Configurações do gráfico
            ax.set_xlabel('Categorias')
            ax.set_ylabel('Lucro/Prejuízo (R$)')
            ax.set_title('Lucro vs Prejuízo por Categoria ')
            ax.axhline(0, color='black', linewidth=1.2)
            ax.grid(axis='y', linestyle='--', linewidth=0.3)
            plt.tight_layout()
            plt.xticks(rotation=25)

            # Visualização Gráfica
            st.subheader("Visualização do Lucro/Prejuízo por Categoria ")
            st.pyplot(fig, use_container_width=True)



                