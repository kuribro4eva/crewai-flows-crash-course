#!/usr/bin/env python
from random import randint
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel
from crews.poem_crew.poem_crew import PoemCrew
from crewai import LLM
import os

import litellm
litellm.verbose = True  # Enable verbose logging

# Set OpenAI API key
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class PoemState(BaseModel):
    sentence_count: int = 1
    poem: str = ""


class PoemFlow(Flow[PoemState]):
    def __init__(self):
        super().__init__()
        # Configure Hugging Face LLM for the flow
        self.llm = LLM(
            model="huggingface/meta-llama/Llama-3.1-70B-Instruct",
            api_key=os.getenv("HUGGING_FACE_API_TOKEN"),
            temperature=0.7,
            max_tokens=3000,
            context_window=8192
        )
        # Set as default LLM for the flow
        self.default_llm = self.llm
        self.function_calling_llm = self.llm  # Ensure function calls use Hugging Face LLM

    @start()
    def generate_sentence_count(self):
        print("Starting flow")
        self.state.sentence_count = randint(2,7)

    @listen(generate_sentence_count)
    def generate_poem(self):
        print("Generating poem")
        poem_crew = PoemCrew()
        poem_crew.llm = self.llm  # Ensure crew uses the same LLM
        poem_crew.function_calling_llm = self.llm

        result = poem_crew.crew().kickoff(
            inputs={
                "sentence_count": self.state.sentence_count,
            }
        )

        print("Poem Generated:")
        print("#"*77)
        print(result.raw)
        self.state.poem = result.raw

    @listen(generate_poem)
    def save_poem(self):
        print("#"*77)
        print("Saving poem")
        with open("poem.txt", "w") as f:
            f.write(self.state.poem)


def kickoff():
    poem_flow = PoemFlow()
    poem_flow.kickoff()


def plot():
    poem_flow = PoemFlow()
    poem_flow.plot()


if __name__ == "__main__":
    kickoff()
