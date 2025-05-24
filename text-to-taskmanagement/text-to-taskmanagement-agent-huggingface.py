import os
import streamlit as st
import traceback

# Import langchain-related libraries
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------
# Prompt Templates
# ---------------------
PROMPT_1 = """
あなたはプロジェクトマネージャーです。以下のメッセージを読み取り、
次のテンプレートに沿って英語で要約してください。
- Subject: <一行で要件を表す件名>
- Task: <具体的に何をするか>
- Background: <依頼の背景・目的を簡潔に>
- Requester: ◯◯◯◯◯ <Leave it blank>
- Assignee: Shohei
- Due: ◯◯◯◯◯ <Leave it blank>
メッセージ:{text}
"""

# ---------------------
# Model Initialization
# ---------------------
def init_models(temperature=0):
    model_1 = ChatGoogleGenerativeAI(temperature=temperature, model="gemini-2.0-flash-lite")
    return model_1

# ---------------------
# Chain Initialization
# ---------------------

def init_chain():
    model_1 = init_models()
    chain_1 = ChatPromptTemplate.from_messages([("user", PROMPT_1)]) | model_1 | StrOutputParser()
    return chain_1


# ---------------------
# Main Streamlit App
# ---------------------
def main():
    st.title("Text to Task Management Sheet")
    chain_1 = init_chain()
    text_input = st.text_area("Paste the original message:", height=150)

    if st.button("Convert"):
        if not text_input.strip():
            st.warning("Please enter some text.")
            return

        try:
            output = chain_1.invoke({"text": text_input})
            result = "".join(list(output))
            st.subheader("Result")
            st.write(result)


        except Exception as e:
            st.error("An error occurred during the process.")
            st.error(traceback.format_exc())

# Standard Streamlit entry point
if __name__ == "__main__":
    main()
