import streamlit as st
import pandas as pd
import openai  # Directly import openai
from langchain.schema import (
    AIMessage,
    HumanMessage,
)

# Load the CSV data
df = pd.read_csv('datasampah1.csv')

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Configure Streamlit page
st.set_page_config(page_title="WasteWiseChatbot")
st.title('WasteWiseChatbot - Chatbot with Dataset Querying')

# Function to query the dataset
def query_dataset(month):
    result = df[df['nama_bulan'].str.contains(month.upper(), na=False)]
    if not result.empty:
        return result
    else:
        return "Data not found for the specified month."

# Define function to generate chatbot responses
def generate_response(input_text):
    input_text_lower = input_text.lower()
    
    if "saldo" in input_text_lower:
        response = "Haloo.. Saldo anda saat ini tidak tersedia dalam dataset ini."
    elif "bayar iuran pln" in input_text_lower:
        response = "Transaksi pembayaran PLN tidak tersedia dalam dataset ini."
    elif "plan investasi" in input_text_lower or "investasi" in input_text_lower:
        response = "Rencana investasi tidak tersedia dalam dataset ini."
    elif "bulan" in input_text_lower:
        month = input_text.split()[-1]
        response = query_dataset(month)
    else:
        response = "Maaf, saya tidak mengerti pertanyaan Anda. Bisa Anda ulangi dengan lebih jelas?"

    # Store the conversation history
    st.session_state['chat_history'].append(HumanMessage(content=input_text))
    st.session_state['chat_history'].append(AIMessage(content=str(response)))
    return response

# Layout for displaying chat history and input form
st.subheader("Conversation History")
for message in st.session_state['chat_history']:
    if isinstance(message, HumanMessage):
        st.text_area("You said:", value=message.content, height=75, key=str(message))
    elif isinstance(message, AIMessage):
        st.text_area("Bot said:", value=message.content, height=75, key=str(message))

st.subheader("Quick Questions")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Check Data for a Month"):
        st.session_state['input_text'] = "Data bulan JULI?"
with col2:
    if st.button("General Query"):
        st.session_state['input_text'] = "How much data is available for this year?"
with col3:
    if st.button("Specific Data Query"):
        st.session_state['input_text'] = "Data bulan OKTOBER?"
with col4:
    if st.button("Another Query"):
        st.session_state['input_text'] = "Is there data for SEPTEMBER?"

with st.form('my_form'):
    text = st.text_area(
        'Ask a question about the dataset:',
        value=st.session_state.get('input_text', 'Type your question here...'),
        height=150
    )
    submitted = st.form_submit_button('Submit')
    if submitted:
        response = generate_response(text)
        st.text_area("Bot's response:", value=response, height=100)
