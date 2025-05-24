import traceback
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

###### dotenv を利用する場合 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


PROMPT = """
You are an AI language model that helps users generate email replies. Given the context of an email conversation, you will create a well-structured, appropriate response based on the provided inputs. The response should match the specified tone and length.

Input:
1. Sender: The person sending the email (e.g., boss, client, etc.)
2. Email Subject: The subject of the email (e.g., About scheduling a meeting)
3. Email Message: The content of the sender's email (e.g., I would like to adjust the time for tomorrow's meeting, are you available in the afternoon?)
4. What you want to say: The desired response (e.g., I am available after 2 PM.)
5. Length: The desired length of the response (e.g., Within 100 characters)

Output:
Generate a reply that addresses the sender's message, incorporates the user's desired response, and maintains a professional tone.

Examples:
 Sender: Client
   Email Subject: About scheduling a meeting
   Email Message: I would like to adjust the time for tomorrow's meeting, are you available in the afternoon?
   What you want to say: I am available after 2 PM.
   Length: 100 characters
   Generated Reply: Dear [Client's Name], Thank you for reaching out. I am available after 2 PM tomorrow for the meeting. Please let me know if this time works for you. Best regards, [Your Name]

Please generate a reply based on the provided inputs.
---
- Sender: {sender},
- Email Subject : {subject},
- Content of the recipient's email:{message},
- What you want to say:{reply},
---
"""

def init_page():
    st.set_page_config(
        page_title="Email Reply AI Agent",
        page_icon="✉️"
    )
    st.header("Email Reply AI Agent ✉️")


def select_model(temperature=0):
    models = ("GPT-4o","GPT-4o-mini", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model_choice = st.radio("Choose a model:", models)
    if model_choice == "GPT-4o":
        return ChatOpenAI(temperature=temperature, model_name="gpt-4o")
    elif model_choice == "GPT-4o-mini":
        return ChatOpenAI(temperature=temperature, model_name="gpt-4o-mini")
    elif model_choice == "Claude 3.5 Sonnet":
        return ChatAnthropic(temperature=temperature, model_name="claude-3-5-sonnet-20240620")
    elif model_choice == "Gemini 1.5 Pro":
        return ChatGoogleGenerativeAI(temperature=temperature, model="gemini-1.5-pro-latest")

def init_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_messages([
        ("user", PROMPT),
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain

def main():
    init_page()
    chain = init_chain()
    if chain:
        sender = st.selectbox("Sender",("Co-worker", "Boss", "Client", "Friend"),key="sender")
        subject = st.text_input("Email Subject (e.g., About scheduling a meeting)", key="subject")
        message = st.text_area("Content of the recipient's email: (e.g., I would like to adjust the time for tomorrow's meeting, are you available in the afternoon?)", key="message")
        reply = st.text_input("What you want to say: (e.g., I am available after 2 PM.)", key="reply")
        if st.button("Generate the Reply"):
            result = chain.stream({"sender": sender, "subject": subject, "message": message, "reply": reply})
            st.write(result)   
      

if __name__ == '__main__':
    main()

# Style adjustments (optional, remove if not needed)
st.markdown(
"""
<style>
/* Custom style adjustments */
.st-emotion-cache-iiif1v { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)