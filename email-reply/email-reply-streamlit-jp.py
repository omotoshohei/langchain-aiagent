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
あなたはユーザーがメール返信を生成するのを助けるAI言語モデルです。メール会話の文脈を踏まえ、提供された入力に基づいて適切な構造の応答を作成します。応答は指定されたトーンと長さに合わせてください。

入力:
1. 送信者: メールを送る人（例: 上司、クライアントなど）
2. メールの件名: メールの件名（例: 会議のスケジュールについて）
3. メールメッセージ: 送信者のメールの内容（例: 明日の会議の時間を調整したいのですが、午後は空いていますか？）
4. あなたが言いたいこと: 希望する返答（例: 2時以降であれば対応可能です。）
5. 長さ: 応答の希望する長さ（例: 100文字以内）

出力:
送信者のメッセージに対処し、ユーザーの希望する返答を取り入れ、プロフェッショナルなトーンを保ちながら返信を生成してください。

例:
 送信者: クライアント
 件名: 会議のスケジュールについて
 メッセージ: 明日の会議の時間を調整したいのですが、午後は空いていますか？
 返したい言葉: 2時以降であれば対応可能です。
 長さ: 100文字
 生成された返信: [クライアント名]様、ご連絡ありがとうございます。明日の会議について、2時以降でご都合がよろしいですね。この時間で問題ないかご確認ください。よろしくお願いいたします。[あなたの名前]

提供された入力に基づいて返信を生成してください。
---
- 送信者: {sender},
- メールの件名: {subject},
- 受信者のメールの内容:{message},
- あなたが言いたいこと:{reply},
---
"""

def init_page():
    st.set_page_config(
        page_title="Eメール返信AIエージェント",
        page_icon="✉️"
    )
    st.header("Eメール返信AIエージェント ✉️")


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
        sender = st.selectbox("送信者",("同僚", "上司", "クライアント", "友人"),key="sender")
        subject = st.text_input("メールの件名（例: 会議のスケジュールについて）", key="subject")
        message = st.text_area("受信者のメールの内容:（例: 明日の会議の時間を調整したいのですが、午後は空いていますか？）", key="message")
        reply = st.text_input("あなたが言いたいこと:（例: 2時以降で対応可能です。）", key="reply")
        if st.button("Submit"):
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