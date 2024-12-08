
# **RAG Chatbot with FAISS and OpenAI**

This project demonstrates the implementation of a Retrieval-Augmented Generation (RAG) chatbot using FAISS for document retrieval and OpenAI's GPT models for response generation. The chatbot answers user queries based on a local dataset (`meta.jsonl`) and ensures that the responses are strictly derived from the provided documents.

---

## **Table of Contents**
- [Overview](#overview)
- [Features](#features)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Code Workflow](#code-workflow)
- [Limitations and Improvements](#limitations-and-improvements)

---

## **Overview**
This chatbot processes a JSONL dataset containing product descriptions, splits the data into smaller chunks, encodes it into vector representations, and indexes the vectors in a FAISS database for efficient similarity-based retrieval. OpenAI's GPT-4 model is then used to generate responses to user queries, ensuring that the responses strictly rely on the provided documents.

---

## **Features**
- **Efficient Document Retrieval:** Uses FAISS for fast and scalable vector searches.
- **Semantic Understanding:** Embeddings generated using `sentence-transformers/all-MiniLM-L6-v2` ensure semantic similarity during searches.
- **Natural Language Generation:** GPT-4 generates human-like responses based on retrieved data.
- **Custom Prompts:** The model is guided by a structured prompt to ensure reliability and prevent hallucinations.
- **Extensibility:** Easily adaptable to different datasets or embedding models.

---

## **Dependencies**
- Python 3.12
- Libraries:
  - `json`
  - `os`
  - `openai`
  - `dotenv`
  - `langchain`
  - `faiss`
  - `sentence-transformers`
- Tools:
  - `FAISS` for vector indexing and retrieval.

---

## **Setup**

### **1. Clone the repository**
```bash
git clone https://github.com/aghiles-saghir/RAG_OpenAI.git
cd RAG_OpenAI
```

### **2. Install dependencies**
Use `pip` to install the required Python libraries.
```bash
pip install -r requirements.txt
```

### **3. Prepare the environment**
- Create an `.env` file in the project root with your OpenAI API key:
```env
OPENAI_API_KEY=your_openai_api_key
```

### **4. Place your dataset**
- Add your dataset as a `.jsonl` file at `./data/meta.jsonl`.

### **5. Run the script**
Execute the main script to preprocess data, create the FAISS index, and test the chatbot:
```bash
python main.py
```

---

## **Usage**

### **1. Dataset Preparation**
- Ensure the dataset is in JSONL format with the key `description` containing the text data.

### **2. Querying the Chatbot**
- Modify the `query` variable in the script with your desired question.

Example:
```python
query = "Description du produit OnePlus 6T"
```

- Run the script to generate the response.

### **3. Output**
The response will be displayed in the terminal, including citations of the relevant passages.

---

## **File Structure**
```
rag-chatbot/
├── data/                   # Directory containing the meta.jsonl dataset.
├── faiss_index/            # Directory for storing the FAISS index (created at runtime).
├── processed_data/         # Directory for storing preprocessed data (created at runtime).
├── openai_integration.py                 # Main script for the RAG chatbot.
├── requirements.txt        # List of required dependencies.
├── .env                    # File containing the OpenAI API key.
```

---

## **Code Workflow**

### **1. Load Dataset**
The `load_data()` function reads the JSONL dataset and extracts descriptions using `extract_descriptions()`.

### **2. Preprocess Data**
Descriptions are split into manageable chunks with `RecursiveCharacterTextSplitter` to improve retrieval performance.

### **3. Generate Embeddings**
Embeddings are created using `sentence-transformers/all-MiniLM-L6-v2`.

### **4. Create FAISS Index**
The chunks are indexed into FAISS for efficient similarity-based retrieval.

### **5. Configure and Query**
- Relevant passages are retrieved using FAISS.
- A structured prompt ensures GPT-4 generates answers based only on retrieved information.

### **6. Response**
OpenAI GPT processes the prompt and generates a user-friendly response.

---

## **Limitations and Improvements**
### **Current Limitations:**
- **Dependency on OpenAI:** The chatbot relies on an internet connection and OpenAI’s API for response generation.
- **Dataset Size:** FAISS performance may degrade with very large datasets.

### **Future Improvements:**
- **Docker Support:** Containerize the application for easier deployment and environment consistency.
- **Multilingual Support:** Extend the embeddings model to support multiple languages.
- **Cloud Indexing:** Explore cloud-based alternatives like Pinecone for larger datasets.

---

## **Contributors**
- Amayas MAHMOUDI
- Aghiles SAGHIR