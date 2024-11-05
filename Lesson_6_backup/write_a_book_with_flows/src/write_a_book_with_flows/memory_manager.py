# Create a new file: memory_manager.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import os
import logging

class BookProjectMemory:
    """
    Manages PDF content loading and processing for the book writing project.
    """
    
    def __init__(self, project_name: str = "book_project", pdf_dir: str = "./pdf"):
        self.project_name = project_name
        self.memory_dir = f"./book_memory/{project_name}"
        self.pdf_dir = Path(pdf_dir)
        
        # Ensure directories exist
        os.makedirs(self.memory_dir, exist_ok=True)
        os.makedirs(f"{self.memory_dir}/rag", exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Store processed chunks
        self.chunks = []
        
        # Store PDF metadata
        self.processed_pdfs = []
        
    def load_pdfs_from_directory(self) -> Dict[str, Dict]:
        """
        Load all PDF files from the specified directory.
        
        Returns:
            Dict[str, Dict]: Dictionary containing metadata for each processed PDF
        """
        if not self.pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {self.pdf_dir}")
            
        processed_files = {}
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {self.pdf_dir}")
            return processed_files
            
        for pdf_path in pdf_files:
            try:
                metadata = self.load_pdf(pdf_path)
                processed_files[str(pdf_path)] = metadata
                self.processed_pdfs.append(metadata)  # Store metadata
                self.logger.info(f"Successfully processed {pdf_path}")
            except Exception as e:
                self.logger.error(f"Error processing {pdf_path}: {str(e)}")
                continue
                
        return processed_files
    
    def load_pdf(self, pdf_path: str | Path) -> Dict:
        """
        Load and process a single PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict: Metadata about the processed PDF
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        # Load PDF
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        chunks = splitter.split_documents(pages)
        
        # Store chunks
        self.chunks.extend(chunks)
        
        # Prepare metadata
        metadata = {
            'pdf_path': str(pdf_path),
            'file_name': pdf_path.name,
            'num_pages': len(pages),
            'num_chunks': len(chunks),
            'processed_date': datetime.now().isoformat()
        }
        
        return metadata
    
    def get_chunks(self) -> List:
        """Get all processed chunks"""
        return self.chunks
    
    def get_processed_pdfs(self) -> List[Dict]:
        """
        Retrieve metadata for all processed PDFs.
        
        Returns:
            List[Dict]: List of metadata dictionaries for processed PDFs
        """
        return self.processed_pdfs
    
    def clear_memory(self) -> None:
        """Clear all stored chunks and reset memory"""
        self.chunks = []
        self.processed_pdfs = []  # Clear PDF metadata
        # Clear files in memory directory
        for path in Path(self.memory_dir).glob("**/*"):
            if path.is_file():
                path.unlink()
        self.logger.info("Memory cleared")