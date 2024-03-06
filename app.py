import os
import json
import time
import random
import tiktoken
import streamlit as st
from openai import OpenAI
from datetime import datetime

secret_password = st.secrets["SECRET_PASSWORD"]

# Options for message while waiting for chat results
buffer_message = [
    "Consulting the coffee grounds...",
    "Asking the nearest crystal ball...",
    "Deciphering ancient hieroglyphs...",
    "Interpreting the tea leaves...",
    "Summoning digital spirits...",
    "Polishing the crystal ball...",
    "Conducting a séance with the cloud...",
    "Bribing the internet gremlins...",
    "Waiting for the hamsters to power up...",
    "Dusting off the old magic wand...",
    "Flipping through the Book of Shadows...",
    "Consulting the stars and planets...",
    "Summoning a genie from the web...",
    "Eavesdropping on the future...",
    "Channeling my inner fortune teller...",
    "Peering into the depths of the code...",
    "Cracking open a fortune cookie...",
    "Engaging the time travel protocol...",
    "Tuning into the AI frequencies...",
    "Warming up the prediction engine..."
]

class ConversationManager:
    """
    Manages conversations with an AI model via the OpenAI API, implementing token budgeting, conversation history tracking, and dynamic persona adjustment.

    Attributes:
        api_key (str): The API key used to authenticate requests to OpenAI.
        client (OpenAI): The OpenAI client initialized with the provided API key.
        base_url (str): The base URL for the OpenAI API. Defaults to 'https://api.openai.com/v1/chat/completions'.
        history_file (str): The filename where the conversation history is saved. If None, a filename is generated with a timestamp.
        default_model (str): The default model used for generating completions. Defaults to 'gpt-3.5-turbo'.
        default_temperature (float): The default sampling temperature for model responses. Defaults to 0.7.
        default_max_tokens (int): The default maximum number of tokens per completion. Defaults to 150.
        token_budget (int): The maximum number of tokens that can be used. Exceeding this budget trims the conversation history.
        system_messages (dict): Predefined system messages corresponding to different assistant personas.
        system_message (str): The current system message, determining the assistant's persona.
        conversation_history (list): The history of the conversation, including both user and assistant messages.

    Methods:
        count_tokens(text): Counts the number of tokens in the given text.
        total_tokens_used(): Calculates the total number of tokens used in the conversation history.
        enforce_token_budget(): Removes the earliest messages to stay within the token budget.
        set_persona(persona): Sets the assistant's persona based on predefined system messages.
        set_custom_system_message(custom_message): Sets a custom system message for the assistant.
        update_system_message_in_history(): Updates the system message in the conversation history.
        chat_completion(prompt, temperature=None, max_tokens=None, model=None): Generates a completion for the given prompt and updates the conversation history.
        load_conversation_history(): Loads the conversation history from the specified history file.
        save_conversation_history(): Saves the current conversation history to the specified file.
        reset_conversation_history(): Resets the conversation history to only include the current system message.
        update_api_key(new_api_key): Updates the API key used for OpenAI requests.
    """
        
    def __init__(self, api_key, base_url="https://api.openai.com/v1/chat/completions", history_file=None, default_model="gpt-3.5-turbo", default_temperature=0.7, default_max_tokens=150, token_budget=4096):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.base_url = base_url
        if history_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.history_file = f"conversation_history_{timestamp}.json"
        else:
            self.history_file = history_file
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.token_budget = token_budget

        self.system_messages = {
            "default_assistant": "You are a helpful assistant.",
            "sassy_assistant": "You are a sassy assistant that is fed up with answering questions.",
            "angry_assistant": "You are an angry assistant that likes yelling in all caps.",
            "thoughtful_assistant": """You are a thoughtful assistant, always ready to dig deeper.
            You ask clarifying questions to ensure understanding and approach problems with a step-by-step methodology.
            """,
            "custom": "Enter your custom system message here."
        }
        self.system_message = self.system_messages["sassy_assistant"]  # Default persona

        self.load_conversation_history()

    def count_tokens(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.default_model)
        except KeyError:
            print(f"Warning: Model '{self.default_model}' not found. Using 'gpt-3.5-turbo' encoding as default.")
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        tokens = encoding.encode(text)
        return len(tokens)

    def total_tokens_used(self):
        try:
            return sum(self.count_tokens(message['content']) for message in self.conversation_history)
        except Exception as e:
            print(f"An unexpected error occurred while calculating the total tokens used: {e}")
            return None
    
    def enforce_token_budget(self):
        try:
            while self.total_tokens_used() > self.token_budget:
                if len(self.conversation_history) <= 1:
                    break
                self.conversation_history.pop(1)
        except Exception as e:
            print(f"An unexpected error occurred while enforcing the token budget: {e}")

    def set_persona(self, persona):
        if persona in self.system_messages:
            self.system_message = self.system_messages[persona]
            self.update_system_message_in_history()
        else:
            raise ValueError(f"Unknown persona: {persona}. Available personas are: {list(self.system_messages.keys())}")

    def set_custom_system_message(self, custom_message):
        if not custom_message:
            raise ValueError("Custom message cannot be empty.")
        self.system_messages['custom'] = custom_message
        self.set_persona('custom')

    def update_system_message_in_history(self):
        try:
            if self.conversation_history and self.conversation_history[0]["role"] == "system":
                self.conversation_history[0]["content"] = self.system_message
            else:
                self.conversation_history.insert(0, {"role": "system", "content": self.system_message})
        except Exception as e:
            print(f"An unexpected error occurred while updating the system message in the conversation history: {e}")

    def chat_completion(self, prompt, temperature=None, max_tokens=None, model=None):
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        model = model if model is not None else self.default_model

        self.conversation_history.append({"role": "user", "content": prompt})

        self.enforce_token_budget()

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=self.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            ai_response = response.choices[0].message.content

            self.conversation_history.append({"role": "assistant", "content": ai_response})
            self.save_conversation_history()

            return ai_response

        except Exception as e:
            error_message = str(e)
            # Check if the error message is related to an incorrect API key
            if "Incorrect API key" in error_message:
                # Use st.error to display a message in the Streamlit interface
                st.error("Failed to authenticate: Incorrect API key provided. Please check your API key and try again.")
            else:
                st.error(f"An error occurred: {error_message}")

            # Log the error for debugging purposes
            print(f"An error occurred while generating a response: {error_message}")
            return None

    
    def load_conversation_history(self):
        try:
            with open(self.history_file, "r") as file:
                self.conversation_history = json.load(file)
        except FileNotFoundError:
            self.conversation_history = [{"role": "system", "content": self.system_message}]
        except json.JSONDecodeError:
            print("Error reading the conversation history file. Starting with an empty history.")
            self.conversation_history = [{"role": "system", "content": self.system_message}]

    def save_conversation_history(self):
        try:
            with open(self.history_file, "w") as file:
                json.dump(self.conversation_history, file, indent=4)
        except IOError as e:
            print(f"An I/O error occurred while saving the conversation history: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while saving the conversation history: {e}")

    def reset_conversation_history(self):
        self.conversation_history = [{"role": "system", "content": self.system_message}]
        try:
            self.save_conversation_history()  # Attempt to save the reset history to the file
        except Exception as e:
            print(f"An unexpected error occurred while resetting the conversation history: {e}")

    def update_api_key(self, new_api_key):
    # def update_api_key(self):
        """
        Update the API key used by the OpenAI client.
        
        Parameters:
        - new_api_key (str): The new API key to use.
        """
        self.api_key = new_api_key
        self.client = OpenAI(api_key=new_api_key)  # Reinitialize the OpenAI client with the new API key

        # Optionally, you might want to log or print a confirmation that the API key has been updated.
        # print("API key updated successfully.")
        # print("API key now:", self.api_key)


# Streamlit code
st.title("Welcome to my AI Chatbot!")
st.header("Enter your name and API key to get started:")

# Initialize ConversationManager
api_key = ''
if 'chat_manager' not in st.session_state:
    # print('API key set to:', api_key)
    st.session_state['chat_manager'] = ConversationManager(api_key)

chat_manager = st.session_state['chat_manager']


# Collect name and API key, store in session state
with st.expander("Enter your name and API key to get started:", expanded=True):
    name = st.text_input("Name:")
    if name not in st.session_state:
        st.session_state['name'] = name
    else:
        name = st.session_state['name']

    # Collect user's API key
    new_api_key = st.text_input("OpenAI API Key:", type="password", key="api_key_input")

    # Check if the API key has changed
    if new_api_key and (not hasattr(st.session_state, 'chat_manager')
                         or new_api_key != st.session_state.chat_manager.api_key):
        # Update the API key in the ConversationManager instance
        if new_api_key == secret_password:
            # Using a secret password to set the API key from Streamlit secrets
            st.session_state.chat_manager.update_api_key(st.secrets["OPENAI_API_KEY"])
        else:
            st.session_state.chat_manager.update_api_key(new_api_key)
        
        # Optionally, show a notification that the API key was updated
        st.success(f"API key updated successfully.")

# Add explainers
with st.expander('Why do I need to provide an API key? :confused:'):
    st.write('''
            Using AI APIs has a very small cost. On an individual scale, it's
            barely noticeable (especially as you get free credits on creating
            an account). However, if hundreds or thousands of people use an app
            the cost can skyrocket.

            Also, there are now bots scanning the internet for unprotected
            API access. They predate on otherwise safe-to-be-free apps and
            use them for bulk AI calls.

            Making your own API key is really easy. Head to 
            [OpenAi's website](https://platform.openai.com/api-keys) (create an
            account if you don't already have one), navigate to API keys, and
            create a new key. Then simply paste it in the above box.

''')
with st.expander('Do you store my API key? Is it safe? :worried:'):
    st.write('''
            I (the developer) have no access to your API key. The app temporarily
            stores it as a [sesion state](https://docs.streamlit.io/library/api-reference/session-state).
            
            Each time you open this app, you create a new session, that's treated
             seperately to any other simultaneous sessions. And when you close the
             tab, the session ends and all information stored in the sesssion
             state is wiped.


             If you're still worried, you can check the source code of the app
             [here](https://github.com/JosiahBeynon/dq-903). You can also create
             a new API key just for this use, then delete it afterwards.
''')

# Create sidebar
with st.sidebar:

    def reset_sliders():
        '''Resets sliders to default'''
        st.session_state['temperature'] = 0.7
        st.session_state['max_tokens'] = 200

    # Initialize session state values if neede
    if 'temperature' not in st.session_state and 'max_tokens' not in st.session_state:
        reset_sliders()

    # Create sliders
    temperature = st.slider("Select model temperature", min_value=0.0, max_value=1.3,
                            step=0.01,key='temperature'
                             )
    max_tokens_per_message = st.slider("Choose max tokens per message", min_value=5, max_value=2000,
                            step=5, key='max_tokens'
                            )
    st.button("Reset sliders", on_click=reset_sliders)

    # Allow user to choose AI feel
    system_message = st.sidebar.selectbox("Choose your AI personality",
                                          ['Default', 'Sassy', 'Angry',
                                           'Thoughtful', 'Custom']
                                          )
    if system_message == 'Default':
        chat_manager.set_persona('default_assistant')
    elif system_message == 'Sassy':
        chat_manager.set_persona('sassy_assistant')
    elif system_message == 'Angry':
        chat_manager.set_persona('angry_assistant')
    elif system_message == 'Thoughtful':
        chat_manager.set_persona('thoughtful_assistant')
    # Allow user input if 'Custom' is selected
    elif system_message == 'Custom':
        custom_message = st.text_area("Custom system message")
        if st.sidebar.button("Build your custom AI personality"):
            # if not custom_message:
            #     st.write("Make sure you've pressed 'Ctrl+Enter' above")
            chat_manager.set_custom_system_message(custom_message)
            placeholder = st.empty()
            placeholder.write(':tada: Personality updated :tada:')
            time.sleep(3)
            placeholder.empty()


    if st.sidebar.button("Reset conversation history", on_click=chat_manager.reset_conversation_history):
        st.session_state['conversation_history'] = chat_manager.conversation_history

# Build chat function
# Initialize conversation history
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = chat_manager.conversation_history

conversation_history = st.session_state['conversation_history']

# Require name and api_key for chat to appear
# if name and api_key:
# Get chat input
user_input = st.chat_input(f'Welcome, {name}. How can I help you?')

# Use chat_manager to get a response. Settings from sidebar
if user_input:
    with st.spinner(random.choice(buffer_message)):
        response = chat_manager.chat_completion(user_input, temperature=temperature, max_tokens=max_tokens_per_message)
        # st.write('trigger response')

# Display the conversation history
for message in conversation_history:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# st.write(conversation_history)