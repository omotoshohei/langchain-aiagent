import traceback
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

###### Use dotenv if available ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please set your environment variables manually.", ImportWarning)
################################################

# Define prompt templates

# 1. Rap Generation Prompt
RAP_GENERATION_PROMPT = """
Generate a rap according to the following structure:
- Length: Exactly 8 lines, with 8 beats per line.
- Rhyme Scheme: Maintain a rhythmic and poetic flow with an ABAB rhyme scheme.
- Content: Reflect the given topic, occupation, and personal message in the lyrics. Start with an introduction, continue with a body that develops the ideas, and end with a strong conclusion.
- Style: Use internal rhyme and multisyllabic rhymes to enhance lyrical complexity.
- Tone: Adjust the tone according to the emotional nature of the personal message, ranging from motivational to introspective as needed.
Each line should smoothly transition to the next, maintaining continuity in theme and rhythm. The 8 lines should form a complete narrative arc.
- Topic: {topic}
- Occupation: {occupation}
- Personal message: {message}
"""

# 2. Feedback Prompt
FEEDBACK_PROMPT = """
Provide feedback on the following rap, with particular focus on:
- Rhyme evaluation: Verify if each line and beat maintains the rhyme scheme. Highlight areas that lack rhyme with examples, and suggest enhancements for good rhymes.
- Rhythm and rhyme consistency.
- Check if the rap maintains exactly 8 lines with 8 beats each.
- Richness and diversity of vocabulary and expression.
- Effectiveness of overall flow and narrative arc.
No need to provide improvement examples directly.
Rap:
{rap}
"""

# 3. Improvement Prompt
IMPROVEMENT_PROMPT = """
Based on the following rap and feedback, improve the rap. When making improvements, focus on:
- Incorporating the suggestions from the feedback.
- Maintaining rhyme in each line and beat.
- Enhancing rhythm, rhyme, and expression to make the lyrics more compelling.
The improved rap should remain 8 lines, with 8 beats per line.
Rap:
{rap}
Feedback:
{feedback}
"""

def init_page():
    st.set_page_config(
        page_title="Rap Generation AI Agent",
        page_icon="🎶"
    )
    st.header("Rap Generation AI Agent 🎶")

def init_models(temperature=1):
    # First model for generating the initial rap
    rap_generator = ChatOpenAI(temperature=temperature, model_name="gpt-4o")

    # Second model for generating feedback
    feedback_model = ChatAnthropic(temperature=0, model_name="claude-3-5-haiku-20241022")

    # Third model for improving the rap
    rap_improver = ChatOpenAI(temperature=temperature, model_name="gpt-4o")

    return rap_generator, feedback_model, rap_improver

def init_chain():
    rap_generator, feedback_model, rap_improver = init_models()
    
    # Set up prompts and output parser for each model
    rap_generation_prompt = ChatPromptTemplate.from_messages([
        ("user", RAP_GENERATION_PROMPT),
    ])
    feedback_prompt = ChatPromptTemplate.from_messages([
        ("user", FEEDBACK_PROMPT),
    ])
    improvement_prompt = ChatPromptTemplate.from_messages([
        ("user", IMPROVEMENT_PROMPT),
    ])
    
    output_parser = StrOutputParser()
    
    # Chain configuration
    rap_generation_chain = rap_generation_prompt | rap_generator | output_parser
    feedback_chain = feedback_prompt | feedback_model | output_parser
    improvement_chain = improvement_prompt | rap_improver | output_parser
    
    return rap_generation_chain, feedback_chain, improvement_chain

def main():
    init_page()
    rap_generation_chain, feedback_chain, improvement_chain = init_chain()
    if rap_generation_chain and feedback_chain and improvement_chain:
        topic = st.text_input("Topic (e.g., Sunday)", key="topic")
        occupation = st.text_input("Your Occupation (e.g., Data Scientist)", key="occupation")
        message = st.text_input("Personal Message (e.g., Prepare for Tomorrow)", key="message")
        
        if st.button("Generate Rap"):
            try:
                # Step 1: Generate Rap
                with st.spinner('Generating rap...'):
                    rap_generator_output = rap_generation_chain.stream({
                        "topic": topic,
                        "occupation": occupation,
                        "message": message
                    })
                    rap_result = ''.join(list(rap_generator_output))
                st.write("### Initial Rap")
                st.write(rap_result)
                
                # Step 2: Generate Feedback
                with st.spinner('Generating feedback...'):
                    feedback_generator_output = feedback_chain.stream({
                        "rap": rap_result
                    })
                    feedback_result = ''.join(list(feedback_generator_output))
                st.write("### Feedback")
                st.write(feedback_result)
                
                # Step 3: Improve Rap
                with st.spinner('Improving rap...'):
                    improved_rap_generator_output = improvement_chain.stream({
                        "rap": rap_result,
                        "feedback": feedback_result,
                        "topic": topic,
                        "occupation": occupation,
                        "message": message
                    })
                    improved_rap_result = ''.join(list(improved_rap_generator_output))
                st.write("### Improved Rap")
                st.write(improved_rap_result)
                
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