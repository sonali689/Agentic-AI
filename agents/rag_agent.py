import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from utils.config_loader import ConfigLoader
from utils.groq_client import GroqClient

class RAGAgent:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.rag_config = config.get_agent_config('rag')
        self.groq_config = config.config['inference']['groq']
        self.vector_store = None
        self.retriever = None
        self.groq_client = GroqClient(
            model=self.groq_config['model'],
            temperature=self.groq_config['temperature'],
            max_tokens=self.groq_config.get('max_tokens', 1024)
        )
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embedding model and text splitter"""
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.rag_config['embedding_model']
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.rag_config['chunk_size'],
            chunk_overlap=self.rag_config['chunk_overlap']
        )
    def get_supported_formats(self):
        """Return supported file formats"""
        return [".pdf", ".txt"]
    def is_ready(self):
        """Check if RAG agent is ready"""
        return self.retriever is not None
    def load_documents(self, course_path: str) -> List[Document]:
        """Load all documents from a course directory"""
        documents = []
        
        for filename in os.listdir(course_path):
            file_path = os.path.join(course_path, filename)
            
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif filename.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        
        print(f"Loaded {len(documents)} documents from {course_path}")
        return documents
    
    def build_knowledge_base(self, course_path: str, course_name: str):
        """Build vector store from course documents"""
        documents = self.load_documents(course_path)
        
        if not documents:
            raise ValueError(f"No documents found in {course_path}")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=f"./vector_stores/{course_name}"
        )
        
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        print(f"Knowledge base built for {course_name}")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system using Groq"""
        if not self.retriever:
            raise ValueError("Knowledge base not built. Call build_knowledge_base first.")
        
        # Retrieve relevant documents
        relevant_docs = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create RAG prompt
        system_message = """You are a helpful study assistant. Answer the user's question based ONLY on the provided context from their course materials. If the answer cannot be found in the context, say so clearly. Provide concise, accurate answers and cite the relevant sources."""
        
        prompt = f"""Based on the following course materials, please answer this question: {question}

Context from course materials:
{context}

Answer:"""
        
        # Get answer from Groq
        answer = self.groq_client.get_response(prompt, system_message)
        
        return {
            "answer": answer,
            "sources": [doc.metadata.get('source', 'Unknown') for doc in relevant_docs],
            "context": relevant_docs
        }