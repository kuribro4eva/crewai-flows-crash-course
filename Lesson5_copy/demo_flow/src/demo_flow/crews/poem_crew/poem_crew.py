from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from ...llm import LLMProvider

@CrewBase
class PoemCrew():
	"""Poem Crew that uses OpenAI, Hugging Face and Claude LLMs"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self):
		super().__init__()
		
		# Initialize LLMs using the provider
		self.openai_4o = LLMProvider.get_openai_4o()
		self.openai_4o_mini = LLMProvider.get_openai_4o_mini()
		self.hf_70B = LLMProvider.get_huggingface_llama_70b()
		self.cl_sonnet = LLMProvider.get_claude_sonnet_latest()

	@agent
	def poem_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['poem_writer'],
			llm=self.openai_4o
		)

	@task
	def write_poem(self) -> Task:
		return Task(
			config=self.tasks_config['write_poem'],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Research Crew"""
		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
		)
