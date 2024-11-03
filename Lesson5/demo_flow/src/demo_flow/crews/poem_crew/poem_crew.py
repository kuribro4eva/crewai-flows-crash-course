from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import os

@CrewBase
class PoemCrew():
	"""Poem Crew that uses both Hugging Face and Claude LLMs"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

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
	def poem_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['poem_writer'],
			llm=self.cl_llm
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
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
		)
