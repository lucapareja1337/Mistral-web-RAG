import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
#from langchain_community import embeddings
from langchain.embeddings import OllamaEmbeddings  
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import validators 

# URL processing 
def process_input(urls, question):
    model_local = Ollama(model="mistral")

    # URLs to List
    urls_list = urls.split("\n")
    valid_urls = [url for url in urls_list if validators.url(url)]  # Filtra apenas URLs válidas
    if not valid_urls:
        return "Nenhuma URL válida fornecida."

    docs = []
    for url in valid_urls:
        try:
            docs.append(WebBaseLoader(url).load())
        except Exception as e:
            st.warning(f"Erro ao carregar a URL {url}: {e}")

    docs_list = [item for sublist in docs for item in sublist]

    # split into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size = 7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)

    # Chunks to vector DB  (Chroma)
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="web-rag",
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    after_rag_template = """Using the following data, answer to what is being asked: {context}
        Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
            | after_rag_prompt
            | model_local
            | StrOutputParser()
    )
    return after_rag_chain.invoke(question)


# Using Streamlit web API
st.title("Web Document Query with Ollama")
st.write("Enter URLs (one per line) and a question to query the documents.")

# Input fields
urls = st.text_area("Enter URLs separated by new lines", height=150)
question = st.text_input("Question")

# Button to process input
if st.button('Query Documents'):
    with st.spinner('Processing...'):
        answer = process_input(urls, question)
        st.text_area("Answer", value=answer, height=300, disabled=True) 
