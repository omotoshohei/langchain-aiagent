import traceback
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

###### If using dotenv ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################

# Define prompt templates

# 1. First Proofread Prompt
FIRST_PROOFREAD_PROMPT = """
Please proofread the following text and check for grammatical and vocabulary accuracy. When proofreading, please pay attention to the following:
- Correct any grammatical errors to ensure the text is accurate.
- Maintain overall consistency and ensure each sentence flows naturally.
- Check that vocabulary choices are appropriate and expressions are clear, improving them if necessary.
- Adjust according to the desired content.
Text: {text}
Desired Content: {user_need}
"""

# 2. Feedback Prompt
FEEDBACK_PROMPT = """
Provide feedback on the following proofreading result and suggest further improvements. When providing feedback, please consider the following:
- Point out any remaining grammatical, vocabulary, or expression issues in the proofread text.
- Suggest specific corrections for areas that need improvement.
- Consider the overall readability and tone.
- Ensure feedback aligns with the desired content.
- Do not suggest any example sentences.
Desired Content: {user_need}
Proofread Result: {proofread_result}
"""

# 3. Improvement Prompt
SECOND_PROOFREAD_PROMPT = """
Please perform a final proofread of the following text, refining the overall tone and nuance to make it natural and easy to read. When proofreading, please pay attention to the following:
- Adjust the overall flow and rhythm of the text to ensure it is smooth and easily understood by the reader.
- Eliminate redundant parts and use concise and clear expressions.
- Choose vocabulary and phrasing that suit the target audience, considering cultural nuances.
- Reflect the feedback to make improvements.
- Make final adjustments to align with the desired content.
Text: {proofread_result}
Desired Content: {user_need}
Feedback: {feedback_result}
"""

def init_page():
    st.set_page_config(
        page_title="Text Proofreading AI Agent",
        page_icon="🎶"
    )
    st.header("Text Proofreading AI Agent 🎶")

# Function to initialize models
def init_models():
    first_proofread_model = ChatAnthropic(temperature=0, model_name="claude-3-5-haiku-20241022")
    feedback_model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    second_proofread_model = ChatAnthropic(temperature=1, model_name="claude-3-5-sonnet-20241022")
    return first_proofread_model, feedback_model, second_proofread_model

def init_chain():
    first_proofread_model, feedback_model, second_proofread_model = init_models()
    
    # Set up prompts and output parser
    first_proofread_prompt = ChatPromptTemplate.from_messages([
        ("user", FIRST_PROOFREAD_PROMPT),
    ])
    feedback_prompt = ChatPromptTemplate.from_messages([
        ("user", FEEDBACK_PROMPT),
    ])
    second_proofread_prompt = ChatPromptTemplate.from_messages([
        ("user", SECOND_PROOFREAD_PROMPT),
    ])
    
    output_parser = StrOutputParser()
    
    # Configure chains
    first_proofread_chain = first_proofread_prompt | first_proofread_model | output_parser
    feedback_chain = feedback_prompt | feedback_model | output_parser
    second_proofread_chain = second_proofread_prompt | second_proofread_model | output_parser
    
    return first_proofread_chain, feedback_chain, second_proofread_chain

def main():
    init_page()
    first_proofread_chain, feedback_chain, second_proofread_chain = init_chain()
    if first_proofread_chain and feedback_chain and second_proofread_chain:
        text = st.text_area("Text to Proofread", key="text")
        user_need = st.text_input("Desired Content (Optional)", key="user_need")
        if st.button("Proofread Text"):
            try:
                # Step 1: Generate Proofread
                with st.spinner('Proofreading text...'):
                    first_proofread_output = first_proofread_chain.stream({
                        "text": text,
                        "user_need": user_need,
                    })
                    first_proofread_result = ''.join(list(first_proofread_output))
                st.write("### First Proofread")
                st.write(first_proofread_result)
                
                # Step 2: Generate Feedback
                with st.spinner('Generating feedback...'):
                    feedback_output = feedback_chain.stream({
                        "proofread_result": first_proofread_result,
                        "user_need": user_need,
                    })
                    feedback_result = ''.join(list(feedback_output))
                st.write("### Feedback")
                st.write(feedback_result)
                
                # Step 3: Improve Proofread
                with st.spinner('Improving text...'):
                    final_text_output = second_proofread_chain.stream({
                        "proofread_result": first_proofread_result,
                        "user_need": user_need,
                        "feedback_result": feedback_result,
                    })
                    final_text = ''.join(list(final_text_output))
                st.write("### Improved Text")
                st.write(final_text)
                
            except Exception as e:
                st.error("An error occurred during processing.")
                st.error(traceback.format_exc())
                
# Style adjustments (optional)
def style_adjustments():
    st.markdown(
    """
    <style>
    /* Custom style adjustments */
    .st-emotion-cache-iiif1v { display: none !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )
    
if __name__ == '__main__':
    main()
    style_adjustments()