import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
from chromadb import PersistentClient

# Use in-memory ChromaDB to bypass SQLite errors
client = PersistentClient(path="/tmp/chroma") 

# Load API Key
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=openai_api_key)

# Streamlit UI
st.title("🚀 LinkedIn Post Generator with AI")
st.write("Generate engaging LinkedIn posts using AI-powered agents!")

# User Inputs
topic = st.text_input("Enter the topic (e.g., AI in Marketing)")
tone = st.selectbox("Select tone", ["Professional", "Engaging", "Storytelling", "Casual"])
audience = st.selectbox("Target Audience", ["Tech Professionals", "Startup Founders", "Marketing Executives"])
post_type = st.selectbox("Post Type", ["Thought Leadership", "Story-based", "Listicle", "Case Study"])

# Define Agents
content_creator = Agent(
    role="Content Creator",
    goal="Generate engaging LinkedIn post ideas and write a compelling post.",
    backstory="A social media strategist with experience in viral LinkedIn posts.",
    llm=llm
)

seo_specialist = Agent(
    role="SEO & Engagement Specialist",
    goal="Optimize LinkedIn posts with proper structure, hashtags, and engagement strategies.",
    backstory="A LinkedIn growth hacker with expertise in content optimization.",
    llm=llm
)

editor = Agent(
    role="Editor & Proofreader",
    goal="Refine LinkedIn posts to be concise, engaging, and professional.",
    backstory="A seasoned copywriter who enhances readability and impact.",
    llm=llm
)

# Define Tasks with expected_output
post_idea_task = Task(
    description=f"Generate 3 LinkedIn post ideas on '{topic}' for {audience}.",
    expected_output="A list of 3 creative LinkedIn post ideas.",
    agent=content_creator
)

post_generation_task = Task(
    description=f"Write a LinkedIn post in a '{tone}' tone for {audience} on '{topic}'.",
    expected_output="A well-structured LinkedIn post (max 300 words).",
    agent=content_creator
)

optimization_task = Task(
    description="Optimize the post with engaging language and hashtags.",
    expected_output="A refined post with added hashtags and improved engagement potential.",
    agent=seo_specialist
)

editing_task = Task(
    description="Proofread and finalize the LinkedIn post before publishing.",
    expected_output="A polished, professional LinkedIn post ready for publishing.",
    agent=editor
)

# Define Crew
crew = Crew(
    agents=[content_creator, seo_specialist, editor],
    tasks=[post_idea_task, post_generation_task, optimization_task, editing_task]
    memory=client
)


# Button to generate the post
if st.button("Generate LinkedIn Post"):
    if topic:
        with st.spinner("Generating your LinkedIn post..."):
            result = crew.kickoff()
        st.success("✅ LinkedIn post generated successfully!")
        st.write("### 📌 Your LinkedIn Post:")
        st.write(result)
        st.code(result, language="markdown")  # Display the post in a copy-friendly format
    else:
        st.warning("⚠️ Please enter a topic to generate a post.")
