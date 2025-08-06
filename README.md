# 🧠 AI Agent for Digital Financial Literacy 💰

This project is an interactive AI-powered assistant designed to educate users about **Digital Financial Literacy** topics such as UPI, online scams, digital wallets, budgeting, loans, and more. Built using **LangChain**, **Milvus Vector DB**, **HuggingFace Embeddings**, and **Replicate's Granite LLM**, the agent provides simple, culturally-inclusive responses to common financial queries.

## 🚀 Features
- 🤖 AI-driven Q&A on Digital Finance topics.
- 📚 Uses a **local vector database (Milvus)** for Retrieval-Augmented Generation (RAG).
- 🧩 Powered by **IBM Granite LLM via Replicate API**.
- 🔗 Simplified explanations of tools like UPI, Wallets, Loans, and Scam Protection.
- 🌐 Gradio Web Interface for easy user interaction.

## 🏗️ Tech Stack
- Python
- Gradio
- LangChain
- Milvus Vector Database
- Replicate API (Granite-3.3-8b-instruct LLM)
- HuggingFace Embeddings (MiniLM-L6-v2)
- Transformers Library

## 📂 Project Structure

├── app.py # Main Python script to run the AI Agent

├── finance_literacy.txt # Knowledge base text file

├── requirements.txt # Python dependencies

└── README.md # This file



## ⚙️ Installation & Setup

git clone https://github.com/yourusername/AI-Agent-Digital-Financial-Literacy.git
cd AI-Agent-Digital-Financial-Literacy
pip install -r requirements.txt

Add your Replicate API Token in app.py:
os.environ['REPLICATE_API_TOKEN'] = "your_replicate_token"

Run the App:
python app.py

📚 Acknowledgements
Mentor: Narendra Eluri

Inspired by Digital Financial Literacy initiatives in India.

Built with LangChain, Replicate, HuggingFace, Milvus, Gradio.

📜 License
MIT License.


