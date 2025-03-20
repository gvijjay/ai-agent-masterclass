import os
import streamlit as st
from crewai import Agent, Task, Crew

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# Load environment variables
api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit UI
st.title("AI Blog Post Generator 📝")
st.write("Enter a topic and let AI generate a well-structured blog post for you!")

# User input
topic = st.text_input("Enter the blog topic:", placeholder="e.g., The Future of AI")

# Button to generate blog post
if st.button("Generate Blog Post"):
    if not topic.strip():
        st.warning("Please enter a valid topic.")
    else:
        with st.spinner("Generating your blog post..."):
            # Define Agents
            researcher = Agent(
                role="Researcher",
                goal="Find relevant information and insights on a given topic.",
                backstory="A seasoned research analyst skilled in gathering precise and useful data.",
                verbose=True,
                llm_model="gpt-4o-mini",
                openai_api_key=api_key 
            )

            writer = Agent(
                role="Writer",
                goal="Write a well-structured and engaging blog post based on research.",
                backstory="An expert content writer who specializes in crafting high-quality blog posts.",
                verbose=True,
                llm_model="gpt-4o-mini",
                openai_api_key=api_key  
            )

            reviewer = Agent(
                role="Reviewer",
                goal="Refine the blog post by correcting errors and improving readability.",
                backstory="A meticulous editor with an eye for detail and clarity.",
                verbose=True,
                llm_model="gpt-4o-mini",
                openai_api_key=api_key  
            )

            # Define Tasks
            research_task = Task(
                description=f"Research the given topic '{topic}' and provide key points.",
                agent=researcher,
                expected_output="A list of 5-10 key points with relevant details."
            )

            writing_task = Task(
                description=f"Write a detailed blog post about '{topic}' based on the research findings.",
                agent=writer,
                expected_output="A structured blog post with an introduction, body, and conclusion."
            )

            review_task = Task(
                description=f"Review and refine the blog post on '{topic}' for grammar, clarity, and structure.",
                agent=reviewer,
                expected_output="A final polished blog post, free of errors and well-structured."
            )

            # Create Crew
            crew = Crew(
                agents=[researcher, writer, reviewer],
                tasks=[research_task, writing_task, review_task],
                llm_model="gpt-4o-mini",
                openai_api_key=api_key  
            )

            # Run CrewAI
            result = crew.kickoff(inputs={"topic": topic})

            # Ensure proper display
            if isinstance(result, list):
                final_output = result[-1]  # Get the last processed result (Reviewed version)
            else:
                final_output = result

            # Display Result in a readable format
            st.subheader("Generated Blog Post:")
            st.markdown(final_output, unsafe_allow_html=True)  # Preserves formatting
