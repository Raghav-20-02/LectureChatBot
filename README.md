# Machine-Learning-Projects

# Lecture Q&A Chatbot 🤖

A Streamlit-powered chatbot that allows you to ask questions about lecture content from a set of web pages. It uses Langchain for natural language processing, Google Generative AI for embeddings and question answering, and FAISS for efficient vector storage and retrieval.

## Features

- **Web Scraping:** Extracts lecture content (headings, text, lists, tables, even MathJax formulas) from given URLs.
- **Structured Data:** Organizes content under headings for easier reference in the chatbot.
- **Semantic Search:** Employs Google Generative AI embeddings for accurate understanding of your questions.
- **Chat Interface:** Streamlit provides an intuitive chat-like experience.
- **Caching:** Uses caching to improve response times for repeated queries.
  

## Installation

1. Create a Virtual Environment (Conda):

   It's recommended to use a virtual environment to keep project dependencies isolated. If you have Conda installed:

   ```bash
   conda create -n lecture-chatbot python=3.10  # Or your desired Python version(3.10 or later is reccomended)
   conda activate lecture-chatbot
   
2. Clone the Repository:
3. 
   ```bash
   git clone https://github.com/Raghav-20-02/Machine-Learning-Projects/tree/main
   cd Machine-Learning-Projects
   Obtain a Google Gemini API Key:

4. Visit Google AI Studio: Go to the Google AI Studio. You will need a Google Cloud Project.
   Get API Key: Click on "Get API key" and follow the prompts. You will either create a new Google Cloud project or select an existing one.
   Copy API Key: Make sure to securely copy the API key as it will be shown only once.

5. Install dependencies
   pip install -r requirements.txt

6. Set Environment Variables:
   
   Create a .env file in the project root directory.
   Add your Google API key in the .env file.

7. Run the chatbot using
   streamlit run app.py


**Technologies Used**
1. Streamlit
2. Langchain
3. Google Generative AI(Google Gemini)
4. FAISS
5. BeautifulSoup4
6. Requests
7. dotenv

   
**How It Works**

Content Extraction: The script fetches lecture content from the provided URLs and structures it.

Chunking and Embedding:  The extracted content is split into manageable chunks and embedded into a vector space using Google Generative AI embeddings.

Vector Storage: A FAISS index is used to efficiently store and retrieve these embeddings.

Question Answering: When you ask a question, the chatbot:

Embeds your question into the same vector space.
Finds relevant chunks using FAISS.
Uses Langchain and Google Generative AI's LLM (Gemini-Pro) to generate an answer based on the retrieved context.


**Important Notes:**
1. The script to scrape the webpage is specially designed to extract the contents from the mentioned URLs only if using other URLs then you will hav to change the     script accordingly.



**Things that can be improved**
1. Make the webscraping part to be more dynamic so that it can work on any provided URL.
2. Improve the extraction and display of mathematical formula currently MathJax is being used to extract the formulas.
3. Improve the embedding process as sometimes the bot is unable to retrive data properly.

