from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

class ResourceAssetAgent:
    def __init__(self, api_key):
        self.llm = OpenAI(temperature=0.2, openai_api_key=api_key)
        self.tools = load_tools(["serpapi"], llm=self.llm)
        self.agent = initialize_agent(
            self.tools, 
            self.llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    
    def collect_resources(self, use_cases):
        resources = {}
        
        for idx, use_case in enumerate(self.parse_use_cases(use_cases["proposed_use_cases"])):
            # Search for datasets
            datasets = self.agent.run(
                f"Find datasets on Kaggle, HuggingFace, or GitHub for implementing this AI use case: {use_case['title']}. " +
                f"Context: {use_case['description']}. Return the URLs and brief descriptions."
            )
            
            # Search for implementation resources
            implementation = self.agent.run(
                f"Find GitHub repositories, tutorials, or papers for implementing {use_case['title']} " +
                f"using GenAI, LLMs, or ML technologies. Return URLs and brief descriptions."
            )
            
            # Suggest GenAI solutions if applicable
            genai_solutions = self.agent.run(
                f"Suggest specific GenAI tools or approaches for implementing {use_case['title']}. " +
                "Focus on document search, automated report generation, or AI-powered chat systems if applicable."
            )
            
            resources[f"use_case_{idx+1}"] = {
                "datasets": datasets,
                "implementation_resources": implementation,
                "genai_solutions": genai_solutions
            }
        
        return resources
    
    def parse_use_cases(self, use_cases_text):
        # This function would parse the text from the use case agent into structured use cases
        # For simplicity, we'll assume it returns a list of dictionaries with 'title' and 'description' keys
        # In a real implementation, this would use the LLM to parse the text properly
        parsed_cases = self.llm.generate(
            [{"text": f"Extract the distinct use cases from this text as a JSON list with title and description fields: {use_cases_text}"}]
        )
        # Convert the response to a Python list
        import json
        return json.loads(parsed_cases.generations[0].text)