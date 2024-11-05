from crewai import LLM
from langchain_openai import ChatOpenAI
import os
from typing import Optional

class LLMProvider:
    """
    A provider class for Language Model configurations.
    
    This class centralizes all LLM configurations and provides easy access
    to different LLM instances with preset configurations.
    """
    
    @staticmethod
    def get_openai_4o(temperature: float = 0.7) -> ChatOpenAI:
        """
        Get OpenAI GPT-4 Turbo Latest configuration
        
        Args:
            temperature (float): Temperature setting for the model (0.0 to 1.0)
            
        Returns:
            ChatOpenAI: Configured OpenAI GPT-4 Turbo instance
        """
        return ChatOpenAI(
            model="chatgpt-4o-latest",
            temperature=temperature,
            # max_tokens=2048,           # Smaller context window
            # top_p=0.9,                 # More focused token sampling
            # frequency_penalty=0.2,      # Slightly higher penalty for repetition
            # presence_penalty=0.2,       # Slightly higher penalty for topic repetition
            # timeout=30,                # Shorter timeout
            # streaming=True,            # Enable streaming responses
            # request_timeout=30,        # Shorter HTTP request timeout
        )
    
    @staticmethod
    def get_openai_4o_mini(temperature: float = 0.9) -> ChatOpenAI:
        """
        Get OpenAI GPT-4 Mini configuration
        
        Args:
            temperature (float): Temperature setting for the model (0.0 to 1.0)
            
        Returns:
            ChatOpenAI: Configured OpenAI GPT-4 Mini instance
        """
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            # max_tokens=2048,           # Smaller context window
            # top_p=0.9,                 # More focused token sampling
            # frequency_penalty=0.2,      # Slightly higher penalty for repetition
            # presence_penalty=0.2,       # Slightly higher penalty for topic repetition
            # timeout=30,                # Shorter timeout
            # streaming=True,            # Enable streaming responses
            # request_timeout=30,        # Shorter HTTP request timeout
        )
    
    @staticmethod
    def get_huggingface_llama_70b(
        temperature: float = 0.9,
        max_tokens: int = 3000
    ) -> LLM:
        """
        Get Hugging Face Llama configuration
        
        Args:
            temperature (float): Temperature setting for the model
            max_tokens (int): Maximum tokens for model output
            
        Returns:
            LLM: Configured Hugging Face LLM instance
        """
        return LLM(
            model="huggingface/meta-llama/Llama-3.1-70B-Instruct",
            api_key=os.getenv("HUGGING_FACE_API_TOKEN"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    @staticmethod
    def get_claude_sonnet_latest(
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> LLM:
        """
        Get Claude 3.5 Sonnet configuration
        
        Args:
            temperature (float): Temperature setting for the model
            max_tokens (Optional[int]): Maximum tokens for model output
            
        Returns:
            LLM: Configured Claude LLM instance
        """
        return LLM(
            model="anthropic/claude-3-5-sonnet-latest",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        ) 