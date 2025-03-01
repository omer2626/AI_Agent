
import os
import openai
import chromadb
from dotenv import load_dotenv
from phi.agent import Agent
from rich.prompt import Prompt
import typer
from typing import Optional

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="product_knowledge")

products = [
    {"Product": "UltraPhone X", "Price": "$799", "Features": "6.5\" OLED display, 128GB storage, 12MP camera", "Colors": "Black, Silver, Blue", "Availability": "In stock", "Shipping": "Standard: $4.99 (3-5 days), Express: $12.99 (1-2 days), Free over $50"},
    {"Product": "SmartWatch Pro", "Price": "$299", "Features": "1.8\" AMOLED, Heart Rate Monitor, GPS", "Colors": "Black, White, Red", "Availability": "In stock", "Shipping": "Standard: $4.99 (3-5 days), Express: $12.99 (1-2 days), Free over $50"},
    {"Product": "LuxeHeadphones", "Price": "$199", "Features": "Noise Cancelling, 30-hour battery, Bluetooth 5.0", "Colors": "Black, Gold, White", "Availability": "In stock", "Shipping": "Standard: $4.99 (3-5 days), Express: $12.99 (1-2 days), Free over $50"},
    {"Product": "Gaming Laptop Z1", "Price": "$1499", "Features": "15.6\" 144Hz display, RTX 4060, 1TB SSD", "Colors": "Black, Gray", "Availability": "Limited stock", "Shipping": "Standard: $9.99 (5-7 days), Express: $19.99 (2-3 days), Free over $100"},
    {"Product": "4K Smart TV", "Price": "$999", "Features": "55\" UHD, HDR10, Dolby Atmos", "Colors": "Black", "Availability": "In stock", "Shipping": "Standard: $19.99 (5-7 days), Express: $39.99 (2-3 days), Free over $500"}
]

# Insert data into ChromaDB
for product in products:
    product_text = f"Product: {product['Product']}, Price: {product['Price']}, Features: {product['Features']}, Colors: {product['Colors']}, Availability: {product['Availability']}, Shipping: {product['Shipping']}"
    collection.add(
        ids=[product['Product']],
        documents=[product_text],
        metadatas=[{"Product": product['Product']}]
    )

def retrieve_query(query):
    """Retrieve relevant product information from ChromaDB."""
    results = collection.query(query_texts=[query], n_results=1)
    if results and results['documents']:
        return results['documents'][0][0]
    return "Sorry, I couldn't find relevant information."

# function to get the response for the user query

def llm_response(prompt):
    """Call OpenAI's LLM API to generate a response."""
    response = openai.ChatCompletion.create(
        model="gpt-4",  
        messages=[{"role": "system", "content": "You are a helpful customer support assistant."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

#build AI Agent using phi agent

def ecommerce_agent(user: str = "user"):
    """Builds and returns the customer support AI agent with OpenAI integration."""
    run_id: Optional[str] = None

    def ask_openai_tool(prompt: str) -> str:
        """Tool function to call OpenAI's API."""
        return llm_response(prompt)

    agent = Agent(
        run_id=run_id,
        user_id=user,
        use_tools=True,
        show_tool_calls=True,
        debug_mode=False,
    )
    
    agent.tools = {"ask_openai": ask_openai_tool}  

    return agent


#final response code where agent uses openAI llm as tool and give the response

def final_response(query):
    """Generate AI response using OpenAI LLM via Phi agent."""
    retrieved_info = retrieve_query(query)  
    agent = ecommerce_agent()  

    prompt = f"Relevant Information: {retrieved_info}\nUser Query: {query}\nAnswer:"
    
    response = agent.tools["ask_openai"](prompt)  

    return response  

def main():
    typer.echo("Welcome to the AI Customer Support Agent!")
    while True:
        query = Prompt.ask("You (type 'exit' to quit)")
        if query.lower() == 'exit':
            break
        response = final_response(query)
        typer.echo(f"AI: {response}")

if __name__ == "__main__":
    typer.run(main)
