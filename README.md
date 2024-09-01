![Logo](https://images.indianexpress.com/2024/02/Gemma.jpg)

# RAG Q&A Chat Bot using Gemma and GROQ API


A RAG (Retrieval-Augmented Generation) application combines large language models (LLMs) with a retrieval system to enhance the generation of responses by accessing relevant external knowledge. In this specific case, the RAG application is developed using the GROQ API, OpenAI embeddings, and is trained on the Gemma model.

Components:
1. GROQ API: The GROQ API is utilized to orchestrate the entire process, handling the interactions between the retrieval system, the language model, and the embeddings. It serves as the backbone, managing the flow of data and ensuring that the application runs efficiently.

2. OpenAI Embeddings: OpenAI's embeddings are used to convert textual data into high-dimensional vectors. These embeddings capture the semantic meaning of the text, enabling the retrieval system to find the most relevant information from a vast knowledge base. The embeddings are crucial for matching user queries with relevant documents or data snippets.

3. Gemma Model: The Gemma model, likely a fine-tuned LLM, is employed to generate responses. This model is trained on a specific dataset to understand the domain or context it is intended to operate in. By combining the Gemma model with the retrieval system, the application can generate more accurate and contextually relevant answers.
## Run Locally

1. Clone this repo into your system.
2. Create virtual environment using the command -

```bash
 conda create -p myenv python==3.9.0 || [OR ANY VERSION]
```
3. Now install all the packages which are listed in requirements.txt

```bash
 pip install -r requirements.txt
```

4. Now Create a .env file and paste your GROQ API KEY as well as OPEN AI API key

5. You can also upload your document as well if you want to train it. I have uploaded attention is all you need pdf.

6. To run on streamlit - 

```bash
    streamlit run app.py
```


## Tech Stack

**Frontend Client:** Streamlit Services

**Model Used:** Gemma with the help of GROQ API

**Embedding Layer used:** Open AI

**Vector Database used:** FAISS

**Dataset Used:** Local Dataset - Attention is all you need.pdf


## Feedback

If you have any feedback or just to say Hi!, please reach out to me at mahobiashubham4@gmail.com
## Live Deployment 

Not Deployed as it is using OPEN AI Embeddings which are paid. For converting text into Vectors.



