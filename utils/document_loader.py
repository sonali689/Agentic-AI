import os
from typing import List, Dict, Any
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader, UnstructuredFileLoader
)
from langchain_core.documents import Document
import markdown
from bs4 import BeautifulSoup

class AdvancedDocumentLoader:
    def __init__(self):
        self.supported_extensions = {
            '.pdf': self._load_pdf,
            '.txt': self._load_text,
            '.docx': self._load_docx,
            '.pptx': self._load_pptx,
            '.md': self._load_markdown,
            '.html': self._load_html
        }
    
    def load_documents(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory"""
        documents = []
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in self.supported_extensions:
                try:
                    loader_func = self.supported_extensions[file_ext]
                    docs = loader_func(file_path)
                    documents.extend(docs)
                    print(f"✓ Loaded {filename}")
                except Exception as e:
                    print(f"✗ Failed to load {filename}: {e}")
            else:
                print(f"? Skipped unsupported file: {filename}")
        
        return documents
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def _load_text(self, file_path: str) -> List[Document]:
        loader = TextLoader(file_path)
        return loader.load()
    
    def _load_docx(self, file_path: str) -> List[Document]:
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()
    
    def _load_pptx(self, file_path: str) -> List[Document]:
        loader = UnstructuredPowerPointLoader(file_path)
        return loader.load()
    
    def _load_markdown(self, file_path: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # Convert markdown to plain text
        html_content = markdown.markdown(markdown_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        plain_text = soup.get_text()
        
        return [Document(
            page_content=plain_text,
            metadata={"source": file_path, "type": "markdown"}
        )]
    
    def _load_html(self, file_path: str) -> List[Document]:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        plain_text = soup.get_text()
        
        return [Document(
            page_content=plain_text,
            metadata={"source": file_path, "type": "html"}
        )]