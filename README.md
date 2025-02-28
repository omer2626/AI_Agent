**AI Customer Support Agent** : 
This AI-powered Customer Support Agent uses ChromaDB for retrieving product information and OpenAI's GPT-4 for generating responses. It is designed to assist customers with inquiries about various products.
**Approach**
**Knowledge Base (ChromaDB)**
The system stores product details in ChromaDB, a vector database optimized for efficient retrieval.
When a user asks about a product, ChromaDB retrieves the most relevant information.
**AI-Powered Responses (OpenAI GPT-4)**
The retrieved product details are then passed to OpenAI's GPT-4 via an API call.
The AI formulates a response in natural language, improving customer interaction.
**Agent with Tools (Phi Agent)**
A Phi Agent is created with an OpenAI tool for answering queries.
The agent retrieves information from ChromaDB and uses OpenAI to generate a response.

**Setup & Instruction**
Step 1 : Clone the Repository or download the zip file
Step 2 : Create a Virtual Environment(venv)
Step 3 : Install All Dependencies
               (pip install -r requirements.txt)

Step 4 : Setup Open AI API

Step 5 : Run the python Script
