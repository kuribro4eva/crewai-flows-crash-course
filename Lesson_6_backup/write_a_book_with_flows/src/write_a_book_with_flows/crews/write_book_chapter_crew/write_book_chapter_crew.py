from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from write_a_book_with_flows.types import Chapter
from write_a_book_with_flows.llm import LLMProvider
from write_a_book_with_flows.tools.pdf_content_tool import PDFContentTool

@CrewBase
class WriteBookChapterCrew:
    """Write Book Chapter Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        super().__init__()
        self.llm_provider = LLMProvider()
        self.openai_4o_mini = self.llm_provider.get_openai_4o_mini()

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            tools=[PDFContentTool()],
            llm=self.openai_4o_mini,
            verbose=True,
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config["writer"],
            tools=[PDFContentTool()],
            llm=self.openai_4o_mini,
            verbose=True,
        )

    @task
    def research_chapter(self) -> Task:
        return Task(
            config=self.tasks_config["research_chapter"],
        )

    @task
    def write_chapter(self) -> Task:
        return Task(
            config=self.tasks_config["write_chapter"],
            output_json=Chapter  # Use output_json with the Pydantic model
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
