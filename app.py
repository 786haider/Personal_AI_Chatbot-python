import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import io
from gtts import gTTS
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# Configure the API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Please set the GOOGLE_API_KEY environment variable in a .env file.")
    st.stop()

# Configure the Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Set page configuration
st.set_page_config(
    page_title="Advanced AI Assistant By Haider Hussain",
    page_icon="🤖",
    layout="wide",
)

# Create or get the Gemini model
def get_gemini_model(vision_required=False):
    # Choose the appropriate model based on the requirement
    if vision_required:
        return genai.GenerativeModel('gemini-1.5-pro-vision')
    else:
        return genai.GenerativeModel('gemini-1.5-pro')

# Function to encode images to base64
def encode_image(image):
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    return image

# Generate response from Gemini with text
def generate_text_response(prompt, chat_history=None):
    model = get_gemini_model()
    
    if chat_history is None or len(chat_history) == 0:
        # If no chat history, start a new conversation
        response = model.generate_content(prompt)
    else:
        # Continue existing conversation
        chat = model.start_chat(history=chat_history)
        response = chat.send_message(prompt)
    
    return response.text

# Generate response for image analysis
def generate_image_response(image, prompt):
    model = get_gemini_model(vision_required=True)
    
    # Process the image
    if isinstance(image, str):  # If image is a file path
        with open(image, "rb") as f:
            image_data = f.read()
    else:  # If image is uploaded through Streamlit
        image_data = encode_image(Image.open(image))
    
    response = model.generate_content([
        prompt,
        {"mime_type": "image/jpeg", "data": image_data}
    ])
    
    return response.text

# Function to analyze uploaded CSV/Excel files
def analyze_data_file(file, question):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        return "Unsupported file format. Please upload a CSV or Excel file."
    
    # Generate statistics about the data
    stats = df.describe().to_string()
    
    # Create a sample visualization based on the first two numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
        plt.xlabel(numeric_cols[0])
        plt.ylabel(numeric_cols[1])
        plt.title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
        
        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_image = Image.open(buf)
        
        # Convert the DataFrame sample and statistics to text
        data_summary = f"Data sample (first 5 rows):\n{df.head().to_string()}\n\nStatistics:\n{stats}"
        
        # Use Gemini to analyze the data
        model = get_gemini_model(vision_required=True)
        response = model.generate_content([
            f"This is a dataset with the following information:\n{data_summary}\n\nUser question: {question}",
            {"mime_type": "image/jpeg", "data": encode_image(plot_image)}
        ])
        
        return response.text, plot_image
    
    # If no numeric columns, just analyze the text data
    data_summary = f"Data sample (first 5 rows):\n{df.head().to_string()}\n\nStatistics:\n{stats}"
    
    prompt = f"This is a dataset with the following information:\n{data_summary}\n\nUser question: {question}"
    response = generate_text_response(prompt)
    
    return response, None

# Text-to-speech function
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang, slow=False)
    
    # Save the audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        temp_file_path = fp.name
        tts.save(temp_file_path)
    
    return temp_file_path

# Initialize session state variables for storing conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "audio_files" not in st.session_state:
    st.session_state.audio_files = []

# App title and header
st.title("Advanced AI Assistant with Gemini")
st.markdown("Multi-modal AI assistant powered by Google's Gemini")

# Two-column layout for main content and sidebar
col1, col2 = st.columns([3, 1])

with col1:
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if "image" in message and message["image"] is not None:
                    st.image(message["image"], caption="Uploaded Image")
                
                if "data_viz" in message and message["data_viz"] is not None:
                    st.image(message["data_viz"], caption="Data Visualization")
                
                st.markdown(message["content"])
                
                # Add audio button for assistant messages
                if message["role"] == "assistant" and "audio_path" in message:
                    with open(message["audio_path"], "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format="audio/mp3")

    # User input area
    user_input_container = st.container()
    with user_input_container:
        # Tabs for different input types
        tabs = st.tabs(["Text"])
        
        with tabs[0]:  # Text tab
            user_prompt = st.chat_input("Ask me anything...")
            
            if user_prompt:
                # Add user message to chat history
                st.session_state.messages.append({
                    "role": "user", 
                    "content": user_prompt,
                    "image": None
                })
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(user_prompt)
                
                # Display assistant thinking indicator
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("Thinking...")
                    
                    # Generate response from the model
                    response = generate_text_response(user_prompt, st.session_state.chat_history)
                    
                    # Update the chat history for context
                    st.session_state.chat_history.append({"role": "user", "parts": [user_prompt]})
                    st.session_state.chat_history.append({"role": "model", "parts": [response]})
                    
                    # Generate audio for the response
                    audio_path = text_to_speech(response)
                    st.session_state.audio_files.append(audio_path)
                    
                    # Update the message placeholder with the response
                    message_placeholder.markdown(response)
                    
                    # Add audio playback
                    with open(audio_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format="audio/mp3")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "audio_path": audio_path,
                    "image": None
                })
                
                # Rerun to update the UI
                st.rerun()
        
        # with tabs[1]:  # Image Analysis tab
        #     st.write("Upload an image for analysis")
        #     uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        #     image_prompt = st.text_input("What would you like to know about this image?", placeholder="Describe this image in detail")
            
        #     analyze_button = st.button("Analyze Image")
            
        #     if analyze_button and uploaded_file is not None and image_prompt:
        #         # Add user message to chat history
        #         st.session_state.messages.append({
        #             "role": "user", 
        #             "content": image_prompt,
        #             "image": uploaded_file
        #         })
                
        #         # Display thinking message
        #         with st.chat_message("assistant"):
        #             message_placeholder = st.empty()
        #             message_placeholder.markdown("Analyzing image...")
                    
        #             # Generate response for the image
        #             response = generate_image_response(uploaded_file, image_prompt)
                    
        #             # Generate audio for the response
        #             audio_path = text_to_speech(response)
        #             st.session_state.audio_files.append(audio_path)
                    
        #             # Update the message placeholder with the response
        #             message_placeholder.markdown(response)
                    
        #             # Add audio playback
        #             with open(audio_path, "rb") as audio_file:
        #                 audio_bytes = audio_file.read()
        #                 st.audio(audio_bytes, format="audio/mp3")
                
        #         # Add assistant response to chat history
        #         st.session_state.messages.append({
        #             "role": "assistant", 
        #             "content": response,
        #             "audio_path": audio_path,
        #             "image": None
        #         })
                
        #         # Rerun to update the UI
        #         st.experimental_rerun()
        
        # with tabs[2]:  # File Analysis tab
        #     st.write("Upload a file for AI analysis")
        #     data_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
        #     file_question = st.text_input("What would you like to know about this data?", 
        #                                   placeholder="Analyze this dataset and provide insights")
            
        #     analyze_data_button = st.button("Analyze Data")
            
        #     if analyze_data_button and data_file is not None and file_question:
        #         # Display thinking message
        #         with st.chat_message("assistant"):
        #             message_placeholder = st.empty()
        #             message_placeholder.markdown("Analyzing data file...")
                    
        #             # Add user message to chat history
        #             file_name = data_file.name
        #             st.session_state.messages.append({
        #                 "role": "user", 
        #                 "content": f"I uploaded a file named '{file_name}' and asked: {file_question}",
        #                 "image": None
        #             })
                    
        #             # Analyze the data file
        #             try:
        #                 result, visualization = analyze_data_file(data_file, file_question)
                        
        #                 # Generate audio for the response
        #                 audio_path = text_to_speech(result)
        #                 st.session_state.audio_files.append(audio_path)
                        
        #                 # Update the message placeholder with the response
        #                 message_placeholder.markdown(result)
                        
        #                 if visualization is not None:
        #                     st.image(visualization, caption="Data Visualization")
                        
        #                 # Add audio playback
        #                 with open(audio_path, "rb") as audio_file:
        #                     audio_bytes = audio_file.read()
        #                     st.audio(audio_bytes, format="audio/mp3")
                        
        #                 # Add assistant response to chat history
        #                 st.session_state.messages.append({
        #                     "role": "assistant", 
        #                     "content": result,
        #                     "audio_path": audio_path,
        #                     "data_viz": visualization,
        #                     "image": None
        #                 })
                    
                #     except Exception as e:
                #         error_message = f"Error analyzing file: {str(e)}"
                #         message_placeholder.markdown(error_message)
                #         st.session_state.messages.append({
                #             "role": "assistant", 
                #             "content": error_message,
                #             "image": None
                #         })
                
                # # Rerun to update the UI
                # st.experimental_rerun()

# with col2:
#     # Sidebar with additional options
#     st.sidebar.title("Options")
    
    # # Text-to-speech language selection
    # st.sidebar.subheader("Text-to-Speech")
    # tts_language = st.sidebar.selectbox(
    #     "TTS Language",
    #     options=[
    #         ("English (US)", "en"),
    #         ("Spanish", "es"),
    #         ("French", "fr"),
    #         ("German", "de"),
    #         ("Italian", "it"),
    #         ("Japanese", "ja"),
    #         ("Korean", "ko"),
    #         ("Chinese", "zh")
    #     [
    #         ("English (US)", "en"),
    #         ("Spanish", "es"),
    #         ("French", "fr"),
    #         ("German", "de"),
    #         ("Italian", "it"),
    #         ("Japanese", "ja"),
    #         ("Korean", "ko"),
    #         ("Chinese", "zh"),
    #         ("Urdu", "ur")  # Add Urdu language option


    #     format_func=lambda x: x[0]
    # )
    
    # Model selection
    st.sidebar.subheader("AI Model Gemini Chatbot By Haider ")
    model_type = st.sidebar.radio(
        "Select Model",
        options=["gemini-1.5-pro", "gemini-1.5-flash"],
        index=0
    )
    
    # Clear conversation button
    if st.sidebar.button("Clear Conversation"):
        # Clean up audio files
        for audio_path in st.session_state.audio_files:
            try:
                Path(audio_path).unlink(missing_ok=True)
            except:
                pass
                
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.audio_files = []
        st.rerun()
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Features")
    st.sidebar.markdown("• Text Chat with AI")
    # st.sidebar.markdown("• Image Analysis")
    # st.sidebar.markdown("• Data File Analysis")
    st.sidebar.markdown("• Text-to-Speech Output")
    
    # # API key input (for demo purposes)
    # st.sidebar.markdown("---")
    # st.sidebar.markdown("## API Configuration")
    # new_api_key = st.sidebar.text_input("API Key (optional)", type="password", 
    #                                help="Enter your Gemini API key here if not set in .env file")
    
    # if new_api_key:
    #     # Update the API key if provided through the UI
    #     genai.configure(api_key=new_api_key)

# Clean up on app close
def cleanup():
    # Remove temporary audio files
    for audio_path in st.session_state.audio_files:
        try:
            Path(audio_path).unlink(missing_ok=True)
        except:
            pass

# Register the cleanup function to be called when the script ends
import atexit
atexit.register(cleanup)
