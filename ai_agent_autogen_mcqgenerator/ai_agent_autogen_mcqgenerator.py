import os
import streamlit as st
import autogen
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Configuration for the LLM (e.g., OpenAI GPT-4)
config_list = [
    {
        "model": "gpt-4",  # or "gpt-3.5-turbo"
        "api_key": openai_api_key,
    }
]

# Define AutoGen Agents
user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",  # No human input during chat; handled by Streamlit
    max_consecutive_auto_reply=1,
    code_execution_config={"use_docker": False},  # Disable Docker
    system_message="You relay user inputs to other agents and return their outputs."
)

mcq_agent = autogen.AssistantAgent(
    name="MCQAgent",
    llm_config={"config_list": config_list},
    system_message="You generate multiple-choice questions (MCQs) based on the given topic."
)

display_agent = autogen.AssistantAgent(
    name="DisplayAgent",
    llm_config={"config_list": config_list},
    system_message="You compile and format the final output from MCQAgent."
)

# Group chat setup
group_chat = autogen.GroupChat(
    agents=[user_proxy, mcq_agent, display_agent],
    messages=[],
    max_round=10
)

group_chat_manager = autogen.GroupChatManager(
    groupchat=group_chat,
    llm_config={"config_list": config_list}
)

# Function to generate MCQs
def generate_mcqs(topic):
    # Define the task for the agents
    task = f"""
    Generate 10 Multiple Choice Questions (MCQs) based on the topic: {topic}.
    Each question should have 4 options and a correct answer.
    
    Format the output as follows:
    
    **MCQs:**
    1. <Question>
       A) <Option 1>
       B) <Option 2>
       C) <Option 3>
       D) <Option 4>
       Correct Answer: <Correct Option>

    2. <Question>
       A) <Option 1>
       B) <Option 2>
       C) <Option 3>
       D) <Option 4>
       Correct Answer: <Correct Option>
    ...
    """
    
    # Start chat with group chat manager
    chat_result = user_proxy.initiate_chat(
        group_chat_manager,
        message=task
    )
    
    # Extract the final output from the chat (assuming DisplayAgent provides it)
    final_output = ""
    for msg in chat_result.chat_history:
        if msg["name"] == "DisplayAgent":
            final_output = msg["content"]
            break
    
    # Fallback if no DisplayAgent output
    if not final_output:
        final_output = "No questions were generated. Please try again."
    
    return final_output

# Streamlit UI
def main():
    st.title("MCQ Generator")
    st.write("Enter a topic, and the app will generate 10 MCQs!")

    # Input field for user topic
    user_topic = st.text_input("Enter a topic:")

    if st.button("Generate MCQs"):
        if user_topic.strip() == "":
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Generating MCQs..."):
                try:
                    questions = generate_mcqs(user_topic)
                    st.markdown(questions)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()