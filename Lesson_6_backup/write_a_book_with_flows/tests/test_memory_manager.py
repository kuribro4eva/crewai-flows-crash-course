import logging
from pathlib import Path
from write_a_book_with_flows.memory_manager import BookProjectMemory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_pdf_extraction():
    """Test PDF extraction and memory storage"""
    
    # Initialize memory manager
    memory = BookProjectMemory(
        project_name="test_project",
        pdf_dir="./pdf"  # Adjust this path if needed
    )
    
    try:
        # Clear any existing memory
        memory.clear_memory()
        logging.info("Cleared existing memory")
        
        # Process PDFs
        processed_files = memory.load_pdfs_from_directory()
        
        # Print processing results
        logging.info("\n=== Processing Results ===")
        if not processed_files:
            logging.warning("No PDFs were processed!")
            return
            
        for pdf_path, metadata in processed_files.items():
            logging.info(f"\nProcessed PDF: {metadata['file_name']}")
            logging.info(f"Number of pages: {metadata['num_pages']}")
            logging.info(f"Number of chunks: {metadata['num_chunks']}")
            logging.info(f"Processing date: {metadata['processed_date']}")
        
        # Verify stored data
        stored_pdfs = memory.get_processed_pdfs()
        logging.info("\n=== Stored Metadata ===")
        for pdf_data in stored_pdfs:
            logging.info(f"\nStored PDF: {pdf_data['file_name']}")
            logging.info(f"Chunks: {pdf_data['num_chunks']}")
        
        # Test RAG storage by checking if documents were stored
        # You might want to add more specific tests based on your needs
        logging.info("\n=== RAG Storage Test ===")
        # Add method to verify RAG storage if needed
        
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        raise
    finally:
        # Clean up
        memory.clear_memory()
        logging.info("\nCleared memory after testing")

if __name__ == "__main__":
    test_pdf_extraction() 