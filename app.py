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
st.title("Welcome to AI Chatbot!")
st.header("Enter your name and API key to get started:")


# Collect name and API key, store in session state
if 'name' not in st.session_state:
    name = st.text_input("Name:")
    st.session_state['name'] = name
else:
    name = st.session_state['name']

if 'api_key' not in st.session_state:
    api_key = st.text_input("API Key:", type="password")
    if api_key == secret_password:
        api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state['api_key'] = api_key
else:
    api_key = st.session_state['api_key']