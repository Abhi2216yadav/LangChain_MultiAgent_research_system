from src.tools.tools import web_search, scrape_url
# from src.tools.tools import scrape_url

output = web_search("What is the latest news about IT sofware jobs in india is this still safe or not")
print(output)

results = scrape_url.invoke({"url": "https://ai.economictimes.com"})
print(results)