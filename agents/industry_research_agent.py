# Using LangChain for web search and information extraction
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

class IndustryResearchAgent:
    def __init__(self, api_key):
        self.llm = OpenAI(temperature=0, openai_api_key=api_key)
        self.tools = load_tools(["serpapi", "llm-math"], llm=self.llm)
        self.agent = initialize_agent(
            self.tools, 
            self.llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    
    def research_company(self, company_name):
        # Research company information
        company_info = self.agent.run(
            f"Research {company_name}. Find their main products, services, industry, and strategic focus areas."
        )
        
        # Research industry trends
        industry_info = self.agent.run(
            f"What industry is {company_name} in? What are the key challenges and opportunities in this industry?"
        )
        
        return {
            "company_details": company_info,
            "industry_context": industry_info
        }