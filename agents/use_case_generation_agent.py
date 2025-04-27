from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

class UseCaseGenerationAgent:
    def __init__(self, api_key):
        self.llm = OpenAI(temperature=0.7, openai_api_key=api_key)
        # Include specialized knowledge on AI applications
        self.tools = load_tools(["serpapi", "llm-math"], llm=self.llm)
        self.agent = initialize_agent(
            self.tools, 
            self.llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    
    def generate_use_cases(self, industry_research):
        # Research AI applications in the industry
        ai_trends = self.agent.run(
            f"What are the current AI and ML applications in the {industry_research['industry_context']} industry? " +
            "Focus on GenAI, LLMs, and ML technologies."
        )
        
        # Generate specific use cases
        use_cases = self.agent.run(
            f"Based on this company information: {industry_research['company_details']} " +
            f"and industry trends: {ai_trends}, generate 5 specific AI use cases that could " +
            "improve their operations, enhance customer satisfaction, and boost efficiency. " +
            "For each use case, include a title, description, and expected business impact."
        )
        
        return {
            "ai_industry_trends": ai_trends,
            "proposed_use_cases": use_cases
        }
