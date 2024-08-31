import streamlit as st
import pandas as pd
import openai

# Set Streamlit page configuration - This must be at the top of the script
st.set_page_config(page_title="WasteWiseChatbot", layout="centered")

# Load CSV data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

data = load_data('datasampah1.csv')

# Display data if necessary
if st.checkbox('Show CSV Data'):
    st.write(data)

# OpenAI API Key (Make sure you set your API key in Streamlit's secrets)
openai.api_key = st.secrets["openai_api_key"]

# Function to generate responses using OpenAI's API
def generate_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "I'm sorry, I couldn't generate a response."

# User input
question = st.text_input("Ask a question based on the document:")

if question:
    # Formulate a prompt to query based on the CSV content
    prompt = f"Based on the following data:\n\n{data.head(5).to_string()}\n\nQ: {question}\nA:"
    
    # Generate a response
    answer = generate_response(prompt)
    
    # Display the answer
    st.write(answer)
