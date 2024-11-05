from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from write_a_book_with_flows.types import BookOutline
from write_a_book_with_flows.llm import LLMProvider
from write_a_book_with_flows.tools.pdf_content_tool import PDFContentTool

@CrewBase
class OutlineCrew:
    """Book Outline Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        super().__init__()
        self.llm_provider = LLMProvider()
        self.openai_4o_mini = self.llm_provider.get_openai_4o_mini()
        self.hf_70b = self.llm_provider.get_huggingface_llama_70b()

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            tools=[PDFContentTool()],
            llm=self.openai_4o_mini,
            verbose=True,
        )

    @agent
    def outliner(self) -> Agent:
        return Agent(
            config=self.agents_config["outliner"],
            tools=[PDFContentTool()],
            llm=self.openai_4o_mini,
            verbose=True,
        )

    @task
    def research_topic(self) -> Task:
        return Task(
            config=self.tasks_config["research_topic"],
        )

    @task
    def generate_outline(self) -> Task:
        return Task(
            config=self.tasks_config["generate_outline"],
            output_json=BookOutline
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            memory=True,
            verbose=True
        )
