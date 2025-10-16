from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
# from langchain_community.chat_models import ChatGroq
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# import os
from dotenv import load_dotenv
load_dotenv()
# api_key = os.getenv("API_KEY")

loader = PyPDFLoader("candidate_resume.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
resume_chunks = splitter.split_documents(docs)

# for chunk in resume_chunks[:2]:
#     print(chunk.page_content[:300])

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS vector DB
vector_db = FAISS.from_documents(resume_chunks, embedding_model)
retriever = vector_db.as_retriever()


# Create FAISS vector DB
vector_db = FAISS.from_documents(resume_chunks, embedding_model)
retriever = vector_db.as_retriever()

job_description = "We are hiring a Full Stack Developer with expertise in AWS, Python, and LangChain."

similar_resumes = retriever.get_relevant_documents(job_description)
# print("Top matching resumes:", [doc.page_content[:100] for doc in similar_resumes])

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

prompt = PromptTemplate.from_template("""
Evaluate this resume against the job description.
Assign a score (0–100) based on skills, experience, and job fit.
Provide a brief summary explaining the score.

Resume: {resume}
Job Description: {job}
""")

chain = LLMChain(llm=llm, prompt=prompt)

resume_text = " ".join([doc.page_content for doc in similar_resumes])
score_response = chain.run({"resume": resume_text, "job": job_description})

print("AI Score:", score_response)
