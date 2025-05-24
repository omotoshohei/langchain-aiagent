import os
import streamlit as st
import traceback

# Import langchain-related libraries
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# If you're using Hugging Face Secrets, you'll do something like:
# os.environ["OPENAI_API_KEY"]     = st.secrets["OPENAI_API_KEY"]
# os.environ["ANTHROPIC_API_KEY"]  = st.secrets["ANTHROPIC_API_KEY"]
# os.environ["GOOGLE_API_KEY"]     = st.secrets["GOOGLE_API_KEY"]

# ---------------------
# Prompt Templates
# ---------------------
PROMPT_1 = """
次のテキストを、{source_language}から{target_language}に翻訳してください。翻訳する際には以下の点に注意してください:
- 元のテキストの意味やニュアンスを忠実に反映してください。
- 自然で流暢な文章になるように心がけてください。
- 特定の用語がある場合は、その用語に対応する一般的な翻訳を使用してください。
- 文化的な違いがある場合は、適切に調整してください。
- 「IKEA」のテキストは「イケア」と表記ください
- Text: {text}
"""

PROMPT_2 = """
次の文章を最終的に校正し、全体のトーンやニュアンスを自然で読みやすい形に整えてください。校正する際には以下の点に注意してください:
- 文章全体の流れやリズムが自然かを確認し、必要に応じて調整してください。
- 簡潔でユーザーにとって有益な情報を伝えるように、冗長な部分を省いてください。
- 語彙や言い回しが適切で、ターゲット読者にとって分かりやすいかをチェックしてください。
- 文化的な背景やニュアンスに配慮し、元の意味が適切に伝わるようにしてください。
- Text: {text}
"""

# ---------------------
# Model Initialization
# ---------------------
def init_models(temperature=0):
    """
    Initialize each model with your chosen parameters.
    Make sure these environment variables are set:
      OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
    """
    model_1 = ChatGoogleGenerativeAI(temperature=temperature, model="gemini-2.0-flash-lite")
    model_2 = ChatAnthropic(temperature=temperature, model_name="claude-3-7-sonnet-latest")
    return model_1, model_2

# ---------------------
# Chain Initialization
# ---------------------
def init_chain():
    # 1. Initialize Models
    model_1, model_2 = init_models()

    # 2. Configure Prompts
    prompt_1 = ChatPromptTemplate.from_messages([
        ("user", PROMPT_1),
    ])
    prompt_2 = ChatPromptTemplate.from_messages([
        ("user", PROMPT_2),
    ])

    # 3. Output Parser
    output_parser = StrOutputParser()

    # 4. Build Simple Chains
    chain_1 = prompt_1 | model_1 | output_parser
    chain_2 = prompt_2 | model_2 | output_parser

    return chain_1, chain_2

# ---------------------
# Main Streamlit App
# ---------------------
def main():
    st.title("Translation & Proofreading Demo")

    # Initialize the chain
    chain_1, chain_2 = init_chain()

    # Collect user input from Streamlit widgets
    source_language = st.selectbox("Source Language", ["English", "Japanese"], index=0)
    target_language = st.selectbox("Target Language", ["English", "Japanese"], index=1)
    text_input = st.text_area("Enter the text you want to translate/proofread:", height=150)

    if st.button("Run Translation & Proofreading"):
        if not text_input.strip():
            st.warning("Please enter some text.")
            return

        try:
            # Step 1: Translation
            step1_output = chain_1.invoke({
                "source_language": source_language,
                "target_language": target_language,
                "text": text_input
            })
            translation_result = "".join(list(step1_output))
            st.subheader("1. Translation Result")
            st.write(translation_result)

            # Step 2: Proofreading
            step2_output = chain_2.invoke({
                "text": translation_result,
            })
            proofread_result = "".join(list(step2_output))
            st.subheader("2. Proofread Result")
            st.write(proofread_result)

        except Exception as e:
            st.error("An error occurred during the process.")
            st.error(traceback.format_exc())

# Standard Streamlit entry point
if __name__ == "__main__":
    main()
