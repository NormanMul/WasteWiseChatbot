import streamlit as st
import pandas as pd
import openai

# Initialize the OpenAI API key
openai_api_key = st.sidebar.text_input('OpenAI API Key', type="password")
if openai_api_key:
    openai.api_key = openai_api_key
else:
    st.warning("Please provide a valid OpenAI API key!")

# Load the CSV data
df = pd.read_csv('datasampah1.csv')

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Configure Streamlit page
st.set_page_config(page_title="WasteWiseChatbot")
st.title('WasteWiseChatbot - Simple Q&A Chatbot')

# Function to search the CSV and generate a response
def search_csv(input_text):
    # Simple search in the CSV file
    matching_rows = df[df.apply(lambda row: input_text.lower() in row.to_string().lower(), axis=1)]
    
    if matching_rows.empty:
        return "I couldn't find any relevant information in the data."

    # Extract some of the matching rows to use in the response
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
            final_response = f"An error occurred: {str(e)}"
    else:
        final_response = "Unable to process your request."
    
    # Store the conversation history
    st.session_state['chat_history'].append({"user": input_text, "bot": final_response})
    return final_response

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
