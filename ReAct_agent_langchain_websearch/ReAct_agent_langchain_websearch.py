import streamlit as st
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from googlesearch import search
import requests
from bs4 import BeautifulSoup


openai_api_key=st.secrets["OPENAI_API_KEY"]

#  Google Search Scraper (No API Key)
def google_search_scraper(query):
    search_results = []
    try:
        for url in search(query, num_results=5):  # Get top 5 links
            search_results.append(url)
    except Exception as e:
        return f"Error: {str(e)}"

    return search_results if search_results else ["No results found."]

#  Web Scraper for First Link
def scrape_first_link(search_results):
    if not search_results or "No results found." in search_results:
        return None

    first_link = search_results[0]
    try:
        response = requests.get(first_link, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs[:5]])  # Extract first 5 paragraphs
        return content if content else None
    except Exception as e:
        return None  # If scraping fails, return None

#  Initialize AI Model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=3000, openai_api_key=openai_api_key)  # Increase max_tokens for longer responses

#  Google Search Tool
def smart_search_tool(query):
    search_results = google_search_scraper(query)
    scraped_content = scrape_first_link(search_results)

    if scraped_content:
        return scraped_content  # Return scraped web content
    else:
        return llm.predict(f"Generate an informative answer for: {query}")  # Fallback to GPT-4

# Define AI Tool
search_tool = Tool(name="Smart Search", func=smart_search_tool, description="Search Google for information.")

#  Initialize ReAct Agent
agent = initialize_agent([search_tool], llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)

#  Streamlit UI
st.title("AI-Powered Google Web search (No API Key Needed)")
st.write("Ask anything, and it will fetch the information.")

query = st.text_input("Enter your query:")
if query:
    with st.spinner("Searching..."):
        response = agent.invoke({"input": query})  
    st.write(response["output"])  #  Extract the final answer
