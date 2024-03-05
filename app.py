import os
import json
import tiktoken
import streamlit as st
from openai import OpenAI
from datetime import datetime

secret_password = st.secrets["SECRET_PASSWORD"]

class ConversationManager:
    def __init__(self, api_key, base_url="https://api.openai.com/v1", history_file=None, default_model="gpt-3.5-turbo", default_temperature=0.7, default_max_tokens=150, token_budget=4096):
        self.client = OpenAI(api_key=api_key)
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
            "sassy_assistant": "You are a sassy assistant that is fed up with answering questions.",
            "angry_assistant": "You are an angry assistant that likes yelling in all caps.",
            "thoughtful_assistant": "You are a thoughtful assistant, always ready to dig deeper. You ask clarifying questions to ensure understanding and approach problems with a step-by-step methodology.",
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
        except Exception as e:
            print(f"An error occurred while generating a response: {e}")
            return None

        ai_response = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        self.save_conversation_history()

        return ai_response
    
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


# Streamlit code
st.title("Welcome to my AI Chatbot!")
st.header("Enter your name and API key to get started:")


# Collect name and API key, store in session state
with st.expander("Enter your name and API key to get started:", expanded=True):
    name = st.text_input("Name:")
    if name not in st.session_state:
        st.session_state['name'] = name
    else:
        name = st.session_state['name']

    api_key = st.text_input("OpenAI API Key:", type="password")
    if api_key == secret_password:
        api_key = st.secrets["OPENAI_API_KEY"]
    if api_key not in st.session_state:
        st.session_state['api_key'] = api_key
    else:
        api_key = st.session_state['api_key']
    
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

# Initialize ConversationManager
if 'chat_manager' not in st.session_state:
    st.session_state['chat_manager'] = ConversationManager(api_key)

chat_manager = st.session_state['chat_manager']

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
    max_tokens = st.slider("Choose max tokens", min_value=5, max_value=2000,
                            value=200, step=5, key='max_tokens'
                            )
    st.button("Reset sliders", on_click=reset_sliders)

    # Allow user to choose AI feel
    system_message = st.sidebar.selectbox("Choose your AI personality",
                                          ['Sassy', 'Angry', 'Thoughtful', 'Custom']
                                          )
    if system_message == 'Sassy':
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
            st.write(':tada: Personality updated :tada:')
            st.write(custom_message)


    if st.sidebar.button("Reset conversation history", on_click=chat_manager.reset_conversation_history):
        st.session_state['conversation_history'] = chat_manager.conversation_history