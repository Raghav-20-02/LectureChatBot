import streamlit as st
from langchain.chains import RetrievalQA
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from typing import List, Union
import faiss


def extract_lecture_content(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the lecture title
    lecture_title_tag = soup.find('span')
    lecture_title = lecture_title_tag.get_text()

    # Extract content, prioritizing the main content div or article tag
    content_div = soup.find('div', {'id': 'main-content', 'class': 'main-content', 'role': 'main'}) or soup.find('article')

    if content_div is None:
        return None, f"Unable to extract content for lecture: {lecture_title}"

    # Extract subheadings (h2 and h3 tags)
    subheadings_h2 = [tag for tag in content_div.find_all('h2')]
    subheadings_h3 = [tag for tag in content_div.find_all('h3')]

    # Organize content under subheadings (including initial title section)
    structured_content = {lecture_title: ""} 
    current_h2 = lecture_title
    current_h3 = None
    in_mathjax_div = False

    for tag in content_div:
        if tag in subheadings_h2:
            current_h2 = tag.get_text()
            structured_content[current_h2] = {} 
            current_h3 = None
            in_mathjax_div = False
        elif tag in subheadings_h3:
            current_h3 = tag.get_text()
            if isinstance(structured_content[current_h2], dict):
                structured_content[current_h2][current_h3] = ""
            else:
                structured_content[current_h2] = {current_h3: ""} 
            in_mathjax_div = False
        else:
            content_to_add = ""
            if tag.name == 'div' and tag.get('class') == ['MathJax']:
                content_to_add = f"`\n{tag.get_text()}\n`\n"
            elif in_mathjax_div:
                in_mathjax_div = False
            elif tag.name in ['p', 'h3']:
                content_to_add = f"{tag.get_text()}\n"
            elif tag.name == 'table':
                content_to_add = str(tag) + "\n"
            elif tag.name == 'ul':
                # Handle unordered lists
                list_items = [f"- {item.get_text()}" for item in tag.find_all('li')]
                content_to_add = "\n".join(list_items) + "\n"

            # Add content to the appropriate level
            if current_h3:
                if not isinstance(structured_content[current_h2][current_h3], str):
                    structured_content[current_h2][current_h3] = "" 
                structured_content[current_h2][current_h3] += content_to_add
            else:
                if isinstance(structured_content[current_h2], dict):
                    structured_content[current_h2] = "" 
                structured_content[current_h2] += content_to_add

    # Prepare the output list
    output_list = [f"Lecture Title: {lecture_title}"]
    for h2_subheading, h3_content in structured_content.items():
        output_list.append(f"\n## {h2_subheading}")
        if isinstance(h3_content, dict):
            for h3_subheading, text in h3_content.items():
                output_list.append(f"\n### {h3_subheading}")
                output_list.append(text.strip())
        else:
            output_list.append(h3_content.strip())

    return output_list



def get_text_chunks(texts: Union[str, List[str]]) -> List[str]:
    """Splits text or a list of texts into chunks using RecursiveCharacterTextSplitter."""

    if isinstance(texts, str):  # Handle single string input
        texts = [texts]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text))
    return chunks


def get_vector_store(chunks, embedding_model="models/embedding-001", index_name="faiss_index"):
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    # try-except block for pickle problem
    try:
        index = faiss.read_index(index_name)
        vectorstore = FAISS(embeddings.embed_query, index, index_to_docstore_id={i: str(i) for i in range(len(chunks))})
    except:
        vectorstore = FAISS.from_texts(chunks, embeddings)
        vectorstore.save_local(index_name) 
    return vectorstore



def get_qa_chain(vectorstore):
    """Sets up the QA chain with a prompt and language model."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not found in the context, say, "Answer not available in the context."

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5, top_p=0.85)
    chain_type_kwargs = {"prompt": prompt}
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), chain_type_kwargs=chain_type_kwargs
    )




def main():
    lecture_urls = [
    "https://stanford-cs324.github.io/winter2022/lectures/introduction/",
    "https://stanford-cs324.github.io/winter2022/lectures/capabilities/",
    "https://stanford-cs324.github.io/winter2022/lectures/harms-1/",
    "https://stanford-cs324.github.io/winter2022/lectures/harms-2/"
]

    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    all_lecture_content = []
    for url in lecture_urls:
        content = extract_lecture_content(url)
        if content:
            all_lecture_content.extend(content)



    st.set_page_config(page_title="Lecture Chatbot", page_icon="🤖")
    st.title("Lecture Q&A Chatbot")
    text_chunks = get_text_chunks(all_lecture_content)
    vectorstore = get_vector_store(text_chunks)  # Load or create the vector store

    # Cache the QA chain (for better performance)
    @st.cache_resource
    def load_chain():
        return get_qa_chain(vectorstore)

    chain = load_chain()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if user_question := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Display bot response with loading indicator
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):  # Show loading indicator
                response = chain.run(user_question)
                st.markdown(response)  # Display the answer
                st.session_state.messages.append({"role": "assistant", "content": response})




if __name__ == "__main__":
    main()
