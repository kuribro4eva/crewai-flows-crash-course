from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
#from langchain_openai import ChatOpenAI

from write_a_book_with_flows.types import Chapter
import os


@CrewBase
class WriteBookChapterCrew:
    """Write Book Chapter Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    #llm = ChatOpenAI(model="gpt-4o")
    
    def __init__(self):
        super().__init__()
        # Configure Hugging Face LLM for all agents
        self.hf_llm = LLM(
            model="huggingface/meta-llama/Llama-3.1-70B-Instruct",
            api_key=os.getenv("HUGGING_FACE_API_TOKEN"),
            temperature=0.9,
            max_tokens=3000,
            #context_window=1000000000000000
            )

        # Configure Claude 3.5 Sonnet LLM
        self.cl_llm = LLM(
            #provider="anthropic",
            model="anthropic/claude-3-5-sonnet-latest",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.7,
            #max_tokens=8192,
        )
    
    @agent
    def researcher(self) -> Agent:
        search_tool = SerperDevTool()
        return Agent(
            config=self.agents_config["researcher"],
            tools=[search_tool],
            llm=self.cl_llm,
            verbose=True,
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config["writer"],
            llm=self.cl_llm,
            verbose=True,
        )

    @task
    def research_chapter(self) -> Task:
        return Task(
            config=self.tasks_config["research_chapter"],
        )

    @task
    def write_chapter(self) -> Task:
        return Task(config=self.tasks_config["write_chapter"], output_pydantic=Chapter)

    @crew
    def crew(self) -> Crew:
        """Creates the Write Book Chapter Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
