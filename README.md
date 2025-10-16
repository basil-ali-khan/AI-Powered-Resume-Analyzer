# AI-Powered Resume Analyzer

An intelligent resume analysis tool that uses AI to evaluate resumes against job descriptions and provide scoring with detailed feedback.

## Features

- **PDF Resume Processing**: Loads and processes PDF resumes using PyPDFLoader
- **Intelligent Text Chunking**: Splits resume content into manageable chunks for better analysis
- **Semantic Search**: Uses HuggingFace embeddings and FAISS vector database for finding relevant resume sections
- **AI-Powered Evaluation**: Leverages Groq's LLaMA model to score resumes against job descriptions
- **Detailed Scoring**: Provides 0-100 scoring with explanatory feedback

## Prerequisites

- Python 3.12+
- Groq API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AI-Powered-Resume-Analyzer
```

2. Install required dependencies:
```bash
pip install langchain langchain-community langchain-groq faiss-cpu pypdf python-dotenv sentence-transformers
```

3. Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

1. Place your resume PDF file in the project directory and name it `candidate_resume.pdf`

2. Run the analyzer:
```bash
python app.py
```

3. The tool will:
   - Load and process the resume
   - Create embeddings and vector database
   - Match resume content against the job description
   - Generate an AI-powered evaluation score

## Project Structure

```
AI-Powered-Resume-Analyzer/
├── app.py                 # Main application file
├── candidate_resume.pdf   # Resume to analyze (add your own)
├── .env                   # Environment variables
└── README.md             # This file
```

## How It Works

1. **Document Loading**: PyPDFLoader extracts text from PDF resumes
2. **Text Splitting**: RecursiveCharacterTextSplitter breaks content into chunks
3. **Embedding Generation**: HuggingFace's all-MiniLM-L6-v2 model creates embeddings
4. **Vector Storage**: FAISS stores embeddings for efficient similarity search
5. **Semantic Matching**: Retriever finds most relevant resume sections for the job
6. **AI Evaluation**: Groq's LLaMA model scores and explains the match

## Customization

- **Job Description**: Modify the `job_description` variable in `app.py`
- **Model Selection**: Change the Groq model in the `ChatGroq` initialization
- **Chunk Size**: Adjust `chunk_size` and `chunk_overlap` parameters
- **Scoring Criteria**: Customize the prompt template for different evaluation criteria

## Dependencies

- `langchain`: Framework for building LLM applications
- `langchain-community`: Community extensions for LangChain
- `langchain-groq`: Groq integration for LangChain
- `faiss-cpu`: Vector similarity search
- `pypdf`: PDF processing
- `python-dotenv`: Environment variable management
- `sentence-transformers`: Embedding models


