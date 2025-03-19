import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()
openai_api_key=st.secrets["OPENAI_API_KEY"]

# Define the state as a plain dictionary (no custom class needed)
def evaluate_options(state: dict):
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key)

    # Construct prompt
    prompt = f"""
    Given the problem: "{state['problem']}",
    Evaluate the following options and choose the best one based on logic, feasibility, and impact:
    {state['options']}
    """

    # Call GPT-4
    response = llm.invoke([{"role": "system", "content": "You are a helpful decision-making assistant."},
                           {"role": "user", "content": prompt}])

    # Return updated state as a dictionary
    return {"problem": state["problem"], "options": state["options"], "evaluation": response.content.strip()}

# Create LangGraph workflow
# Use a dict as the state instead of a custom class
graph = StateGraph(dict)  # State is now a dictionary
graph.add_node("evaluate", evaluate_options)
graph.set_entry_point("evaluate")
graph.add_edge("evaluate", END)
app = graph.compile()

# Streamlit UI
st.title("AI Decision-Making Assistant")
problem = st.text_input("Enter your problem statement:")
options = st.text_area("Enter possible options (comma separated):")

if st.button("Evaluate Decision"):
    if problem and options:
        option_list = [opt.strip() for opt in options.split(",")]
        # Initialize state as a dictionary, not a DecisionState object
        initial_state = {"problem": problem, "options": option_list, "evaluation": ""}
        result = app.invoke(initial_state)
        st.subheader("Decision Analysis:")
        st.write(result["evaluation"])
    else:
        st.warning("Please enter both a problem statement and options.")