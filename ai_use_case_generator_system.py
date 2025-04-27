from langchain.llms import OpenAI
from agents.industry_research_agent import IndustryResearchAgent
from agents.use_case_generation_agent import UseCaseGenerationAgent
from agents.resource_asset_agent import ResourceAssetAgent

class AIUseCaseGeneratorSystem:
    def __init__(self, api_key):
        self.industry_agent = IndustryResearchAgent(api_key)
        self.use_case_agent = UseCaseGenerationAgent(api_key)
        self.resource_agent = ResourceAssetAgent(api_key)
        self.openai = OpenAI(temperature=0.2, openai_api_key=api_key)
    
    def generate_proposal(self, company_or_industry):
        # Step 1: Research the industry/company
        print("Researching company/industry...")
        industry_research = self.industry_agent.research_company(company_or_industry)
        
        # Step 2: Generate use cases
        print("Generating AI use cases...")
        use_cases = self.use_case_agent.generate_use_cases(industry_research)
        
        # Step 3: Collect resource assets
        print("Collecting implementation resources...")
        resources = self.resource_agent.collect_resources(use_cases)
        
        # Step 4: Generate final proposal
        proposal = self.create_final_proposal(
            company_or_industry,
            industry_research,
            use_cases,
            resources
        )
        
        return proposal
    
    def create_final_proposal(self, company, research, use_cases, resources):
        # Format all data into a comprehensive proposal
        proposal_prompt = f"""
        Create a comprehensive AI implementation proposal for {company} with the following information:
        
        Company/Industry Information:
        {research['company_details']}
        
        Industry Context:
        {research['industry_context']}
        
        AI Industry Trends:
        {use_cases['ai_industry_trends']}
        
        Proposed Use Cases:
        {use_cases['proposed_use_cases']}
        
        Implementation Resources:
        {resources}
        
        Format this as a professional proposal with the following sections:
        1. Executive Summary
        2. Company and Industry Analysis
        3. Top AI Use Cases (with business impact)
        4. Implementation Resources (with clickable links)
        5. Next Steps
        
        For each use case, include references to how it was identified and relevant resource links.
        """
        
        proposal = self.openai.generate([{"text": proposal_prompt}])
        return proposal.generations[0].text
    
    def save_proposal(self, proposal, company_name, format="markdown"):
        # Save the proposal to a file
        clean_name = company_name.replace(" ", "_").lower()
        file_name = f"{clean_name}_ai_proposal.md" if format == "markdown" else f"{clean_name}_ai_proposal.txt"
        
        with open(file_name, "w") as f:
            f.write(proposal)
        
        return file_name