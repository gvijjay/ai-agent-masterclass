import streamlit as st
import langgraph.graph as lg 
from langchain_openai import ChatOpenAI  
from langchain.schema import SystemMessage
import os
from typing import TypedDict, Optional
from dotenv import load_dotenv

load_dotenv()
api_key = st.secrets["OPENAI_API_kEY"]

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# State representation
class DebugState(TypedDict):
    code: str
    error: Optional[str]
    fix_suggestion: Optional[str]
    alternative_fixes: Optional[str]

# Function to extract Python errors
def extract_error_message(code):
    try:
        exec(code)  # Runs code (unsafe for production)
    except Exception as e:
        return str(e)
    return "No error detected"

# Node to detect errors
def error_detection(state: DebugState):
    error_message = extract_error_message(state["code"])
    return {**state, "error": error_message}  

# Node to analyze error and suggest a fix
def generate_fix(state: DebugState):
    if state["error"] == "No error detected":
        return {**state, "fix_suggestion": "No errors detected!"}

    prompt = f"""
    Here is a Python code snippet:
    ```
    {state["code"]}
    ```
    It throws the following error:
    ```
    {state["error"]}
    ```
    Please fix the code and explain why the fix works.
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    return {**state, "fix_suggestion": response.content}  

# Node to suggest alternative fixes
def generate_alternative_fix(state: DebugState):
    if state["error"] == "No error detected":
        return {**state, "alternative_fixes": "No alternative fix needed."}

    prompt = f"""
    Here is a Python code snippet:
    ```
    {state["code"]}
    ```
    It throws the following error:
    ```
    {state["error"]}
    ```
    Please suggest an alternative way to fix this issue.
    """
    response = llm.invoke([SystemMessage(content=prompt)])
    return {**state, "alternative_fixes": response.content}  

# Build StateGraph
workflow = lg.StateGraph(DebugState)
workflow.add_node("detect_error", error_detection)  
workflow.add_node("suggest_fix", generate_fix)  
workflow.add_node("suggest_alternative_fix", generate_alternative_fix)  

workflow.set_entry_point("detect_error")
workflow.add_edge("detect_error", "suggest_fix")
workflow.add_edge("suggest_fix", "suggest_alternative_fix")

debugger_agent = workflow.compile()

# Streamlit UI
st.title("🛠️ AI Debugging Companion")
st.write("Enter your Python code below, and the AI will analyze and suggest fixes.")

code_input = st.text_area("Paste your Python code here:", height=200)

if st.button("Debug Code"):
    if code_input.strip():
        result = debugger_agent.invoke({"code": code_input})

        if result["error"] == "No error detected":
            st.success("✅ No errors detected in your code!")
        else:
            st.error(f"🔴 Error detected: {result['error']}")

        st.subheader("✅ Fix Suggestion:")
        st.code(result["fix_suggestion"], language="python")

        st.subheader("🔄 Alternative Fix:")
        st.code(result["alternative_fixes"], language="python")
    else:
        st.warning("⚠️ Please enter some code before debugging.")
