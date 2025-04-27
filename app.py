import streamlit as st
from ai_use_case_generator_system import AIUseCaseGeneratorSystem

st.set_page_config(page_title="AI Use Case Generator", layout="wide")

st.title("AI & GenAI Use Case Generator")
st.write("Generate AI use cases and implementation resources for any company or industry")

# API key input
api_key = st.text_input("Enter your OpenAI API Key", type="password")

# Company/Industry input
company = st.text_input("Enter Company or Industry Name")

if st.button("Generate AI Use Cases") and api_key and company:
    with st.spinner("Generating AI use cases and resources..."):
        try:
            generator = AIUseCaseGeneratorSystem(api_key)
            proposal = generator.generate_proposal(company)
            file_name = generator.save_proposal(proposal, company)
            
            st.success(f"Proposal generated and saved as {file_name}")
            
            # Display the proposal
            st.markdown("## Generated AI Proposal")
            st.markdown(proposal)
            
            # Provide download link
            st.download_button(
                label="Download Proposal",
                data=proposal,
                file_name=file_name,
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"Error generating proposal: {str(e)}")