from duckduckgo_search import DDGS
from langchain.tools import Tool

def search_duckduckgo(query, max_results=3):
    """
    Performs a DuckDuckGo search and returns structured results.
    """
    results = DDGS().text(query, max_results=max_results)
    return [{"title": r["title"], "link": r["href"], "snippet": r.get("body", "")} for r in results]

def search_web(query: str) -> str:
    """Search the web using DuckDuckGo and return formatted results."""
    results = search_duckduckgo(query, max_results=3)
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(f"{i}. {result['title']}\n   Link: {result['link']}\n   Summary: {result['snippet']}")
    return "\n\n".join(formatted_results)

duckduckgo_search_tool = Tool(
    name="Web Search",
    description="Search the web for current information using DuckDuckGo. Input should be a search query string.",
    func=search_web
)
