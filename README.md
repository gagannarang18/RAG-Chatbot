# 🤖 RAG Chatbot Assignment

Build a **Retrieval-Augmented Generation (RAG)** based chatbot using LangChain, powered by **Groq LLM** and **AWS Titan Embeddings**.

---

## 🚀 Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/gagannarang18/RAG-Chatbot.git
cd your-repo-name

# 2. Install required dependencies
pip install -r requirements.txt
```

### 🔐 Environment Variables

Create a `.env` file in the root directory and add the following keys:

```env
GROQ_API_KEY=your_groq_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
```

### ▶️ Run the App

```bash
streamlit run app.py
```

---

## ✨ Features

- ⚙️ Retrieval-Augmented Generation pipeline using LangChain
- 📄 Supports data loading from **CSV**, **JSON**, **PDF**, and **TXT** files
- ⚡️ Fast inference with **Groq LLM**
- 🌐 Embedding powered by **Amazon Titan via AWS**

---

## 🧠 Example Use Case

Ask contextual questions about your uploaded data and get accurate answers using real-time retrieval.

> 💬 *“What is the average revenue in 2022?”*  
> 💬 *“Summarize key points from the product policy.”*

---

## 📸 Demo

![Chatbot Screenshot](assets/demo.png)

---

## 📁 Project Structure

```
├── app.py                  # Streamlit UI
├── rag_chatbot.py         # RAG logic and LangChain setup
├── requirements.txt
├── .env                    # Environment secrets
├── data/                   # Your knowledge base
└── assets/
    └── demo.png            # Demo screenshot
```

---

## 📝 License

MIT License © 2025 Gagan Narang
