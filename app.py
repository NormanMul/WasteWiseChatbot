import streamlit as st
import pandas as pd
import openai

# Initialize the OpenAI API key
openai_api_key = st.sidebar.text_input('OpenAI API Key', type="password")
if openai_api_key:
    openai.api_key = openai_api_key
else:
    st.warning("Please provide a valid OpenAI API key!")

# Load the CSV data with error handling
try:
    df = pd.read_csv('datasampah1.csv')
except FileNotFoundError:
    st.error("The CSV file was not found. Please make sure 'datasampah1.csv' is in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the CSV file: {str(e)}")
    st.stop()

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Configure Streamlit page
st.set_page_config(page_title="WasteWiseChatbot")
st.title('WasteWiseChatbot - CSV-based Q&A Chatbot')

# Function to search the CSV and generate a response
def search_csv(input_text):
    matching_rows = df[df.apply(lambda row: input_text.lower() in row.to_string().lower(), axis=1)]
    
    if matching_rows.empty:
        return "I couldn't find any relevant information in the data."

    # Combine the relevant rows for context
    result = "\n\n".join(matching_rows.head(3).apply(lambda row: row.to_string(), axis=1))
    
    return result

def generate_response(input_text):
    # Use the search function to retrieve data
    retrieved_data = search_csv(input_text)
    
    # Generate a response using OpenAI's GPT
    if openai_api_key and retrieved_data:
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"Based on the following data, answer the question:\n\nData:\n{retrieved_data}\n\nQuestion: {input_text}\n\nAnswer:",
                max_tokens=150,
                temperature=0.7
            )
            final_response = response.choices[0].text.strip()
        except Exception as e:
            final_response = f"An error occurred while communicating with OpenAI: {str(e)}"
    else:
        final_response = "Unable to process your request. Make sure you have entered a valid OpenAI API key and there is relevant data."

    # Store the conversation history
    st.session_state['chat_history'].append({"user": input_text, "bot": final_response})
    return final_response

# Layout for displaying chat history and input form
st.subheader("Conversation History")
for chat in st.session_state['chat_history']:
    st.text_area("You said:", value=chat['user'], height=75, key=f"user_{chat['user']}")
    st.text_area("Bot said:", value=chat['bot'], height=75, key=f"bot_{chat['bot']}")

with st.form('my_form'):
    text = st.text_area(
        'Ask a question:',
        value=st.session_state.get('input_text', ''),
        height=150
    )
    submitted = st.form_submit_button('Submit')
    if submitted:
        response = generate_response(text)
        st.text_area("Bot's response:", value=response, height=100, key="response_area")
