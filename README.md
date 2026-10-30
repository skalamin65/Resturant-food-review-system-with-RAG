# 🍽️ Restaurant Food Review System Using RAG

## 📌 Overview

The **Restaurant Food Review System** is an AI-powered application that uses **Retrieval-Augmented Generation (RAG)** to answer user questions about restaurant menus, food items, customer reviews, and dining experiences.

The system integrates a Large Language Model (LLM) through **Ollama** and uses **LangChain** to manage document processing, retrieval, and response generation. Instead of relying only on the LLM's pre-trained knowledge, the application retrieves relevant information from a restaurant knowledge base and uses it to generate contextual, informative answers.

This project demonstrates how RAG can be applied to build an intelligent, domain-specific conversational AI system.

## ✨ Features

- **Retrieval-Augmented Generation (RAG):** Retrieves relevant information from restaurant documents before generating answers.
- **LLM Integration:** Uses locally hosted language models through Ollama.
- **LangChain Integration:** Manages document loading, text splitting, retrieval, and LLM interaction.
- **Restaurant Menu Analysis:** Answers questions about available food items and menu details.
- **Food Review Analysis:** Uses customer review documents to provide relevant insights about food quality and dining experiences.
- **Semantic Retrieval:** Finds relevant document chunks based on the user's query using vector embeddings.
- **Context-Aware Responses:** Generates answers grounded in the retrieved restaurant information.
- **Local Model Execution:** Supports running an LLM locally through Ollama, depending on the selected model and available hardware.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | RAG pipeline orchestration |
| Ollama | Local LLM integration |
| Vector Store | Stores and retrieves document embeddings |
| Embedding Model | Converts text into vector representations |
| TextLoader | Loads restaurant text documents |
| RecursiveCharacterTextSplitter | Splits documents into manageable chunks |
| RAG | Grounds generated responses in retrieved information |

## 🏗️ System Architecture

The application follows a typical RAG pipeline:

1. **Document Loading:** Loads restaurant menus and food review documents from text files.
2. **Text Splitting:** Divides documents into smaller chunks for efficient retrieval.
3. **Embedding Generation:** Converts text chunks into vector representations using an embedding model.
4. **Vector Storage:** Stores the embeddings and associated text in a vector store.
5. **User Query:** Accepts a question about restaurant food or reviews.
6. **Retrieval:** Searches the vector store for relevant document chunks.
7. **Context Augmentation:** Combines the retrieved information with the user's query.
8. **LLM Generation:** Uses the Ollama-integrated LLM to generate a context-aware response.

### Workflow

```text
Restaurant Menu & Review Files
              |
              v
       Document Loading
              |
              v
         Text Splitting
              |
              v
       Embedding Model
              |
              v
          Vector Store
              |
              v
         User Question
              |
              v
       Relevant Retrieval
              |
              v
      Context + User Query
              |
              v
       LangChain + Ollama
              |
              v
       Generated Answer
```

## 📂 Project Structure

The following is an example structure. Adjust the filenames to match your actual repository.

```text
Resturant-food-review-system-with-RAG/
│
├── Data/
│   └── txt/
│       ├── restaurant_menu.txt
│       └── food_reviews.txt
│
├── embedding_maneger.py
├── vector_store.py
├── rag_retriver.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/skalamin65/Resturant-food-review-system-with-RAG.git
```

Navigate to the project directory:

```bash
cd Resturant-food-review-system-with-RAG
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

If your project includes a `requirements.txt` file, run:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies required by your implementation. These may include:

```bash
pip install langchain langchain-community langchain-text-splitters langchain-ollama chromadb
```

Install any additional embedding-model dependencies used by your project.

### 4. Install Ollama

Download and install Ollama from its official website:

https://ollama.com/

Start Ollama and download a compatible language model. For example:

```bash
ollama pull llama3.1
```

Make sure the model you download matches the model name configured in your Python code.

### 5. Configure the LLM

If your code uses `ChatOllama`, configure it with the selected model:

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1",
    temperature=0.1
)
```

Ensure Ollama is running and the selected model is available locally before launching the application.

### 6. Run the Application

Execute the Python entry-point file. For example:

```bash
python main.py
```

Replace `main.py` with your actual entry-point filename if it is different.

## 💬 Example Queries

Users can ask questions such as:

- Which food items are available on the restaurant menu?
- What do customers think about the quality of the food?
- Which dishes receive positive customer feedback?
- What complaints do customers mention in their reviews?
- What can you tell me about the restaurant's dining experience?
- Summarize the feedback about a particular dish.

The answers depend on the information available in the restaurant documents and the relevance of the retrieved context.

## 🧠 What I Learned

Through this project, I explored and implemented concepts related to:

- Retrieval-Augmented Generation (RAG)
- Large Language Model (LLM) integration
- LangChain components and pipeline orchestration
- Document loading and text chunking
- Text embeddings and vector-based retrieval
- Context-aware prompt construction
- Local LLM inference using Ollama
- Modular Python development for AI applications

## 🚀 Future Improvements

- Develop an interactive interface using Streamlit.
- Integrate voice input and text-to-speech responses.
- Improve retrieval quality using metadata filtering and reranking.
- Add support for PDF menus and additional restaurant documents.
- Evaluate answer relevance, faithfulness, and retrieval performance.
- Deploy the application for remote access.
- Add support for multiple restaurants and larger review datasets.

## 👨‍💻 Author

**Sk Alamin Hossain**

BTech in Computer Science Engineering | BSc (Hons) in Data Science & AI

**Interests:** Artificial Intelligence, Generative AI, Large Language Models, RAG Systems, and Applied AI Engineering.

## 📄 License

This project is intended for educational and portfolio purposes. Add a license file to specify the terms under which others may use, modify, or distribute the project.
