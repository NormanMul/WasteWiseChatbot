import streamlit as st
import pandas as pd
import openai

# Load the CSV data
df = pd.read_csv('datasampah1.csv')

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Configure Streamlit page
st.set_page_config(page_title="WasteWiseChatbot")
st.title('WasteWiseChatbot - Chatbot with OpenAI Integration')

# OpenAI API key configuration
openai_api_key = st.sidebar.text_input('OpenAI API Key', type="password")
temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7, step=0.1)

# Function to query the dataset
def query_dataset(month):
    result = df[df['nama_bulan'].str.contains(month.upper(), na=False)]
    if not result.empty:
        return result
    else:
        return "Data not found for the specified month."

# Function to generate a response using OpenAI's API
def generate_response(input_text):
    input_text_lower = input_text.lower()
    
    if "bulan" in input_text_lower:
        month = input_text.split()[-1]
        response = query_dataset(month)
    else:
        if not openai_api_key:
            response = "Please provide a valid OpenAI API key."
        else:
            # Interact with OpenAI's API
            openai.api_key = openai_api_key
            try:
                response = openai.chat_completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": input_text},
                    ],
                    temperature=temperature,
                ).choices[0].message['content'].strip()
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
