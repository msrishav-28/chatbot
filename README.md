# RAG Chatbot with Groq & Pinecone

## 📌 Overview
This project is a Retrieval-Augmented Generation (RAG) chatbot built using Streamlit, Groq AI, and Pinecone. The chatbot allows users to upload documents (PDF/TXT), retrieve relevant information, and generate responses based on document context using Groq's language models.

## 🚀 Features
- **Document Upload & Processing**: Supports PDF and TXT files, extracts text, and stores vector embeddings in Pinecone.
- **Intelligent Retrieval**: Finds relevant document chunks based on user queries.
- **AI-Powered Responses**: Uses Groq models (Mixtral, LLaMA, Gemma, etc.) for generating responses.
- **Chat History & Persistence**: Saves chat sessions and uploaded documents for continuity.
- **Logging & Debugging**: Captures logs for monitoring chatbot performance.

## 🛠️ Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip

### Clone the Repository
```sh
git clone https://github.com/your-repo/rag-chatbot.git
cd rag-chatbot
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Set Up Environment Variables
Create a `.env` file in the project root with the following:
```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=your_pinecone_environment
```

## 🔄 Usage

### Run the Chatbot
```sh
streamlit run app.py
```

### Upload Documents
- Navigate to the **Document Management** section in the sidebar.
- Upload a PDF or TXT file.
- The file is processed, and embeddings are stored in Pinecone.

### Start Chatting
- Type your query in the chat input box.
- The chatbot retrieves relevant document chunks and generates responses.

### Manage Chat History
- View past chat sessions in the **Chat History** section.
- Start a new chat anytime.
- Download chat logs for reference.

### Clearing Data
- Use the "🗑️ Clear Documents & Index" button to delete all documents and reset Pinecone.

## 🤖 Available AI Models
The chatbot supports the following Groq models:
- `mixtral-8x7b-32768`
- `llama2-70b-4096`
- `llama3-8b-8192`
- `llama3-70b-8192`
- `gemma-7b-it`
- `deepseek-r1-distill-llama-70b`

Users can select their preferred model from the dropdown menu.

## 📝 Logs & Debugging
- Logs are saved in `chatbot.log`.
- The **View Logs** expander in the UI shows recent logs for debugging.

## 📜 License
This project is licensed under the MIT License. Feel free to use and modify it as needed!

## 🙌 Contributing
Pull requests are welcome! If you have suggestions or bug reports, please open an issue.

## 📬 Contact
For questions or support, reach out via msrshav28@gmail.com
