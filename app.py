import streamlit as st
import pandas as pd
import openai
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import SimpleCSVLoader
from langchain.text_splitter import CharacterTextSplitter

# Load the CSV data and prepare it for LangChain
df = pd.read_csv('datasampah1.csv')
df.to_csv('prepared_datasampah.csv', index=False)

# Load CSV into a LangChain-compatible format
loader = SimpleCSVLoader("prepared_datasampah.csv")
documents = loader.load()

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
split_docs = text_splitter.split_documents(documents)

# Initialize the OpenAI LLM with the API key
openai_api_key = st.sidebar.text_input('OpenAI API Key', type="password")
llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)

# Set up the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=loader.as_retriever())

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Configure Streamlit page
st.set_page_config(page_title="WasteWiseChatbot")
st.title('WasteWiseChatbot - Chatbot with LangChain and RAG')

# Function to generate a response using LangChain
def generate_response(input_text):
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
        if openai_api_key.startswith('sk-'):
            response = generate_response(text)
            st.text_area("Bot's response:", value=response, height=100)
        else:
            st.warning('Please enter a valid OpenAI API key!', icon='⚠️')
