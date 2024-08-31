import streamlit as st
import pandas as pd
import openai

# Set Streamlit page configuration
st.set_page_config(page_title="WasteWiseChatbot", layout="centered")

# Load CSV data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

data = load_data('datasampah1.csv')

# Display data if necessary
if st.checkbox('Show CSV Data'):
    st.write(data)

# Sidebar for API key input and temperature slider
st.sidebar.title("Configuration")
api_key_input = st.sidebar.text_input(
    "OpenAI API Key", type="password", key="api_key"
)
temperature = st.sidebar.slider(
    'Temperature', min_value=0.0, max_value=1.0, value=0.7, step=0.1
)

# Store the API key in the session state
if api_key_input:
    st.session_state["openai_api_key"] = api_key_input

# Use the API key from session state
api_key = st.session_state.get("openai_api_key", None)

if not api_key:
    st.sidebar.error("Please enter your OpenAI API key to continue.")
    st.stop()

openai.api_key = api_key

# Function to generate responses using OpenAI's API
def generate_response(prompt, temperature=0.7):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "I'm sorry, I couldn't generate a response."

# User input
question = st.text_input("Ask a question based on the document:")

if question:
    # Formulate a prompt to query based on the CSV content
    prompt = f"Based on the following data:\n\n{data.head(5).to_string()}\n\nQ: {question}\nA:"
    
    # Generate a response
    answer = generate_response(prompt, temperature)
    
    # Display the answer
    st.write(answer)
