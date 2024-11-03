from crewai.flow.flow import Flow, listen, start
from dotenv import load_dotenv
from litellm import completion
import os

load_dotenv()


class ExampleFlow(Flow):
    def __init__(self):
        super().__init__()
        # Just store the model name and API key
        self.model = "huggingface/meta-llama/Llama-3.1-70B-Instruct"
        self.api_key = os.getenv("HUGGING_FACE_API_TOKEN")

    @start()
    def generate_city(self):
        print("Starting flow")

        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "Return the name of a random city in the world.",
                },
            ],
            api_key=self.api_key
        )

        random_city = response["choices"][0]["message"]["content"]
        print(f"Random City: {random_city}")

        return random_city

    @listen(generate_city)
    def generate_fun_fact(self, random_city):
        print("Received random city:", random_city)
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"Tell me a fun fact about {random_city}",
                },
            ],
            api_key=self.api_key
        )

        fun_fact = response["choices"][0]["message"]["content"]
        return fun_fact


flow = ExampleFlow()
result = flow.kickoff()

print(f"Generated fun fact: {result}")
