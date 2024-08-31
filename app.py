import streamlit as st
import pandas as pd
import openai
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.llms import OpenAI

# Initialize the OpenAI API key
openai_api_key = st.sidebar.text_input('OpenAI API Key', type="password")
if openai_api_key:
    openai.api_key = openai_api_key
else:
    st.warning("Please provide a valid OpenAI API key!")

# Load the CSV data
df = pd.read_csv('datasampah1.csv')

# Prepare the documents for LangChain
documents = [Document(page_content=row.to_string()) for _, row in df.iterrows()]

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)

# Initialize OpenAI Embeddings and Vector Store with error handling
if openai_api_key:
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = Chroma.from_documents(split_docs, embeddings)
    except Exception as e:
        st.error(f"An error occurred while initializing embeddings: {str(e)}")
        embeddings = None
else:
    embeddings = None

# Initialize the OpenAI LLM with the API key
if embeddings:
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    qa_chain = load_qa_chain(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
else:
    st.stop()  # Stop the app if embeddings are not initialized

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Configure Streamlit page
st.set_page_config(page_title="WasteWiseChatbot")
st.title('WasteWiseChatbot - Chatbot with LangChain and RAG')

# Function to generate a response using LangChain
def generate_response(input_text):
    if not qa_chain:
        return "QA chain is not properly initialized."
    
    try:
        response = qa_chain.run(input_text)
    except Exception as e:
        response = f"An error occurred: {str(e)}"
    
    # Store the conversation history
    st.session_state['chat_history'].append({"user": input_text, "bot": response})
    return response

# Layout for displaying chat history and input form
st.subheader("Conversation History")
for chat in st.session_state['chat_history']:
    st.text_area("You said:", value=chat['user'], height=75)
    st.text_area("Bot said:", value=chat['bot'], height=75)

with st.form('my_form'):
    text = st.text_area(
        'Ask a question:',
        value=st.session_state.get('input_text', 'Type your question here...'),
        height=150
    )
    submitted = st.form_submit_button('Submit')
    if submitted:
        response = generate_response(text)
        st.text_area("Bot's response:", value=response, height=100)
