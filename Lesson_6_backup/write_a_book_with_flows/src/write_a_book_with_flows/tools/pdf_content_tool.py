from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from crewai_tools import PDFSearchTool
from pathlib import Path
import logging
import shutil

logger = logging.getLogger(__name__)

class PDFContentToolArgs(BaseModel):
    """Arguments for PDFContentTool"""
    query: str = Field(description="The search query to find relevant content in PDFs")

class PDFContentTool(PDFSearchTool):
    """Tool for accessing PDF content"""
    
    name: str = "pdf_content_tool"
    description: str = "Search through PDF content using semantic search"
    args_schema: Type[BaseModel] = PDFContentToolArgs
    
    def __init__(self, **kwargs):
        pdf_dir = Path("./pdf")
        cache_dir = Path("./.cache")
        
        # Clear existing cache
        if cache_dir.exists():
            logger.info("Clearing existing cache...")
            shutil.rmtree(cache_dir)
            logger.info("Cache cleared")
        
        if not pdf_dir.exists():
            raise ValueError(f"PDF directory not found at {pdf_dir}")
            
        pdfs = list(pdf_dir.glob("*.pdf"))
        if not pdfs:
            raise ValueError("No PDF files found in ./pdf directory")
            
        logger.info(f"Found {len(pdfs)} PDF files: {[p.name for p in pdfs]}")
        logger.info("Creating new embeddings...")
        
        # Initialize with fresh embeddings
        super().__init__(
            pdf=str(pdfs[0]),
            config=dict(
                llm=dict(
                    provider="openai",
                    config=dict(
                        model="gpt-4-turbo-preview",
                        temperature=0.7
                    )
                ),
                embedder=dict(
                    provider="openai",
                    config=dict(
                        model="text-embedding-3-small"  # Using smaller model to match dimensions
                    )
                )
            )
        )
        logger.info("Embeddings created successfully")
        
    def _run(self, query: str) -> str:
        """Override run to add logging"""
        try:
            logger.debug(f"Searching PDFs with query: {query}")
            result = super()._run(query)
            logger.debug("Search completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error searching PDFs: {e}")
            raise