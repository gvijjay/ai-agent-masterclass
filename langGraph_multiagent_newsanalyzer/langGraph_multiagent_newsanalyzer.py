import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph
from typing import TypedDict

# Load API keys
load_dotenv()
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

# Initialize OpenAI model with GPT-4o-mini
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# Define News API Fetcher
def fetch_news(topic):
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={NEWS_API_KEY}&pageSize=4&sortBy=publishedAt&language=en"
    response = requests.get(url)
    articles = response.json().get("articles", [])

    news_data = []
    for article in articles:
        title = article.get("title")
        description = article.get("description")
        link = article.get("url")
        published_at = article.get("publishedAt")

        # Skip articles that don't have a title or description
        if title and description:
            news_data.append({"title": title, "description": description, "link": link, "published_at": published_at})
    
    return news_data
 

# Define State for Multi-Agent Workflow
class AgentState(TypedDict):
    news: str
    summary: str
    fake_news: str
    sentiment: str

# Summarization Agent
def summarizer(state):
    return {"summary": llm.predict(f"Summarize this article: {state['news']}")}

# Fake News Detection Agent
def fake_news_detector(state):
    return {"fake_news": llm.predict(f"Detect if this news contains fake or misleading information: {state['news']}")}

# Sentiment Analysis Agent
def sentiment_analyzer(state):
    return {"sentiment": llm.predict(f"Analyze sentiment: {state['summary']}")}

# Build LangGraph Multi-Agent Workflow
workflow = StateGraph(AgentState)
workflow.add_node("summarizer", summarizer)
workflow.add_node("fake_news_detector", fake_news_detector)
workflow.add_node("sentiment_analyzer", sentiment_analyzer)

workflow.set_entry_point("summarizer")
workflow.add_edge("summarizer", "fake_news_detector")
workflow.add_edge("summarizer", "sentiment_analyzer")

runnable = workflow.compile()

# Streamlit UI
st.title("📰 AI News Analyzer (Multi-Agent)")

topic = st.text_input("Enter a topic (e.g., AI, Sports, Economy)")
if st.button("Analyze News"):
    news_list = fetch_news(topic)

    if not news_list:
        st.error("No articles found.")
    elif len(news_list) < 4:
        st.warning(f"Only {len(news_list)} articles found for '{topic}'. Some might be missing required fields.")

    for i, news in enumerate(news_list):
        title = news.get("title", "No Title Available")
        description = news.get("description", "No Description Available")

        st.subheader(f"Article {i+1}: {title}")
        st.write(f"📅 Published On: {news.get('published_at', 'Unknown')[:10]}")
        st.write(f"**Description:** {description}")
        st.write(f"🔗 [Read Full Article]({news.get('link', '#')})")  

        # Process AI Analysis
        news_text = f"{title} - {description}"
        result = runnable.invoke({"news": news_text})

        st.write(f"**Summary:** {result.get('summary', 'No Summary Available')}")
        st.write(f"**Fake News Check:** {result.get('fake_news', 'No Fake News Check Available')}")
        st.write(f"**Sentiment Analysis:** {result.get('sentiment', 'No Sentiment Analysis Available')}")


