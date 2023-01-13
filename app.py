# Bibliotecas
import pandas as pd
import time
import joblib
import base64
from io import StringIO
import streamlit as st
import re
import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout='wide')

# configurações
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Base de Dados
dados = pd.read_csv('SentimentalAnalytics/modelo_oceane_instagram_posts_comments-limpo.csv', sep=';', error_bad_lines=False)

# Removendo dados nulos
dados = dados.dropna()

# Removendo dados duplicados
dados.drop_duplicates(['commentId'], inplace=True)

df = dados.copy()

# Limpeza dos comentários
def limpeza_dados(texto):

    # Retirar caracteres especiais
    limpeza_1 = re.sub( '["("")"?@|$|.|!,:%;"]', '', texto)

    # Conveter para minusculo
    limpeza_2 = limpeza_1.lower()

    # Retirar Hastgas
    limpeza_3 = re.sub( '#\S+', '', limpeza_2)

    # Retirar valores numericos
    limpeza_4 = re.sub('[0-9]', '', limpeza_3)

    # Retirar Emojis
    eliminar_emojis = re.compile(
        "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF" 
            u"\U0001F1E0-\U0001F1FF"  
            u"\U0001F1F2-\U0001F1F4"  
            u"\U0001F1E6-\U0001F1FF" 
            u"\U0001F600-\U0001F64F"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U0001f970"
            u"\U0001F1F2"
            u"\U0001F1F4"
            u"\U0001F620"
            u"\u200d"
            u"\u2640-\u2642"
        "]", flags=re.UNICODE
        )

    limpeza_5 = eliminar_emojis.sub('', limpeza_4)
    return limpeza_5

df['comment'] = df['comment'].apply(limpeza_dados)


# StopWords
#nltk.download('stopwords')
grupo_palavras = nltk.corpus.stopwords.words('portuguese')

def remover_stopwords (texto):
    lista_palavras = texto.split()
    frase_ajustada = ''

    for loop in lista_palavras:
        if loop not in grupo_palavras:
            frase_ajustada = frase_ajustada + ' ' + loop

    return frase_ajustada

df['comment'] = df['comment'].apply(remover_stopwords)


# Extração de Radical
#nltk.download('rslp')
radical = nltk.stem.RSLPStemmer()

def extracao_radical (texto):

    lista_palavras = texto.split()
    frase_ajustada = ''

    for loop in lista_palavras:
        extracao = radical.stem(loop)
        frase_ajustada = frase_ajustada + ' ' + extracao

    return frase_ajustada

df['comment'] = df['comment'].apply(extracao_radical)


# Modelo Sentimento
X = df['comment']
y = df['sentimento']

X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Instanciar o metodo
token = TweetTokenizer()
vectorizer = CountVectorizer(analyzer='word', tokenizer=token.tokenize)

# Aplicando a transformaação
X_train_1 = vectorizer.fit_transform(X_train)
type(X_train_1)

# Aplicando o modelo
model_lr = joblib.load('SentimentalAnalytics/modelo_lr.sav')
model_lr.fit(X_train_1, y_train)

X_test_1 = vectorizer.transform(X_test)
y_pred = model_lr.predict(X_test_1)


# Aplicação do Modelo
def funcao_sentimento(texto):

  limpeza = limpeza_dados(texto)

  stopwords = remover_stopwords(limpeza)

  radical = extracao_radical(stopwords)

  vetor = vectorizer.transform([radical])

  previsao_lr = model_lr.predict(vetor)[0]

  return previsao_lr


# Modelo para contexto
X_cont = df['comment']
y_cont = df['contexto']

X_train_cont, X_test_cont, y_train_cont , y_test_cont = train_test_split(X_cont, y_cont, test_size=0.2, random_state=42)

# Aplicando a transformaação
X_train_2 = vectorizer.fit_transform(X_train_cont)
type(X_train_2)

# Aplicando o modelo
model_lr_cont = joblib.load('SentimentalAnalytics/modelo_lr_cont.sav')
model_lr_cont.fit(X_train_2, y_train_cont)

X_test_cont_2 = vectorizer.transform(X_test_cont)
y_pred_cont = model_lr_cont.predict(X_test_cont_2)


# Aplicação do Modelo
def funcao_contexto(texto):

  limpeza = limpeza_dados(texto)

  stopwords = remover_stopwords(limpeza)

  radical = extracao_radical(stopwords)

  vetor = vectorizer.transform([radical])

  previsao_lr_cont = model_lr_cont.predict(vetor)[0]

  return previsao_lr_cont

# Titulo
st.title('Sentimental Analytics')

# Menu lateral
menu_lateral = st.sidebar.empty()
uploaded_file = st.sidebar.file_uploader('Choose a file')

# Upload do arquivo 
def uploaded():
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df_upload = pd.read_csv(uploaded_file)

    if not uploaded_file:
        st.warning('Please upload is a file for analytics.')
        st.stop()
    with st.spinner('Wait for it...'):
        time.sleep(5)
    st.success('Done!')
    return df_upload

# Download aqurivo final
def get_download(df_upload, arq):
    csv = df_upload.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() 
    df_final = f'<a href="data:file/csv;base64,{b64}" download="'+arq+'.csv">Download data as CSV</a>'
    return df_final


def main():

    df_upload = uploaded()

    # Removendo dados duplicados
    df_upload.drop_duplicates(['commentId'], inplace=True)

    # Aplicando a função de sentimento
    df_upload.insert(4, 'sentimento', df_upload['comment'].apply(funcao_sentimento), allow_duplicates=False)

    # Aplicando a função de contexto
    df_upload.insert(5, 'contexto', df_upload['comment'].apply(funcao_contexto), allow_duplicates=False)

    # Reply Oceane
    df_upload.loc[df_upload['username'] == 'oceane', 'sentimento'] = 'Neutro'

    df_upload.loc[df_upload['username'] == 'oceane', 'contexto'] = 'Resposta da marca'

    st.write(df_upload)

    # Download arquivo final
    st.markdown(get_download(df_upload, 'df_final'), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
