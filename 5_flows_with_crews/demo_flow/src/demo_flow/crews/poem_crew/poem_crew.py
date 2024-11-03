from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import os

@CrewBase
class PoemCrew:
    """Poem Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        super().__init__()
        # Configure Hugging Face LLM for all agents
        self.llm = LLM(
            model="huggingface/meta-llama/Llama-3.1-70B-Instruct",
            api_key=os.getenv("HUGGING_FACE_API_TOKEN"),
            temperature=0.9,
            max_tokens=3000,
            context_window=8192
        )

    @agent
    def poem_writer(self) -> Agent:
        """Agent that writes the initial poem"""
        return Agent(
            config=self.agents_config["poem_writer"],
            llm=self.llm,
            function_calling_llm=self.llm  # Ensure function calls use Hugging Face LLM
        )

    @task
    def write_poem(self) -> Task:
        return Task(
            config=self.tasks_config["write_poem"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Poem Writing Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,a
            llm=self.llm,
            function_calling_llm=self.llm  # Ensure function calls use Hugging Face LLM
        )
