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

# 1. ラップ生成プロンプト
RAP_GENERATION_PROMPT = """
下の内容をテーマにした、8小節16行の日本語ラップを書いてください。
- 豊富な語彙で韻を踏んでください。
- トピック: {topic}
- 職業: {occupation}
- 個人的なメッセージ: {message}
"""

# 2. フィードバックプロンプト
FEEDBACK_PROMPT = """
8小節16行のラップになっていることを確認して、間違ってたら16行目以降を省略して。
四行ごとに[Verse 1][Verse 2][Verse 3][Verse 4]と見出しをつけて。
- ラップ：{rap}
"""

# # 3. 改善プロンプト
# IMPROVEMENT_PROMPT = """
# 生成したラップを改善ください。以下の内容を盛り込んで。
# - 出来るだけ多く韻を含めて。
# - 言葉遊びやメタファーも積極的に取り入れてください。
# - ８行８拍子はキープ。
# - ラップ：{feedback}
# """

def init_page():
    st.set_page_config(
        page_title="ラップ生成AIエージェント",
        page_icon="🎶"
    )
    st.header("ラップ生成AIエージェント 🎶")

def init_models(temperature=1):
    # 最初のモデルでラップを生成
    rap_generator = ChatGoogleGenerativeAI(temperature=temperature, model="gemini-exp-1114")

    # rap_generator = ChatOpenAI(temperature=1, model_name="gpt-4o-mini")
    # rap_generator = ChatAnthropic(temperature=temperature, model_name="claude-3-5-sonnet-20241022")

    # 2番目のモデルでフィードバックを生成
    feedback_model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    # feedback_model = ChatOpenAI(temperature=temperature, model_name="gpt-4o-mini")
    # feedback_model = ChatAnthropic(temperature=temperature, model_name="claude-3-5-haiku-20241022")

    # # 3番目のモデルでラップを改善
    # rap_improver = ChatOpenAI(temperature=0.7, model_name="gpt-4o")
    # # rap_improver = ChatAnthropic(temperature=temperature, model_name="claude-3-5-haiku-20241022")


    return rap_generator, feedback_model

def init_chain():
    rap_generator, feedback_model = init_models()
    
    # 各プロンプトと出力パーサーの設定
    rap_generation_prompt = ChatPromptTemplate.from_messages([
        ("user", RAP_GENERATION_PROMPT),
    ])
    feedback_prompt = ChatPromptTemplate.from_messages([
        ("user", FEEDBACK_PROMPT),
    ])
    
    output_parser = StrOutputParser()
    
    # チェーンの構成
    rap_generation_chain = rap_generation_prompt | rap_generator | output_parser
    feedback_chain = feedback_prompt | feedback_model | output_parser
    
    return rap_generation_chain, feedback_chain

def main():
    init_page()
    rap_generation_chain, feedback_chain = init_chain()
    if rap_generation_chain and feedback_chain:
        topic = st.text_input("トピック（例：日曜日）", key="topic")
        occupation = st.text_input("あなたの職業（例：データサイエンティスト）", key="occupation")
        message = st.text_input("伝えたいメッセージ（例：明日に備える）", key="message")
        if st.button("ラップを生成する"):
            try:
                # ステップ1: ラップ生成
                with st.spinner('ラップを生成中...'):
                    rap_generator_output = rap_generation_chain.stream({
                        "topic": topic,
                        "occupation": occupation,
                        "message": message
                    })
                    rap_result = ''.join(list(rap_generator_output))
                st.write("### 初期ラップ")
                st.write(rap_result)
                
                # ステップ2: フィードバック生成
                with st.spinner('フィードバックを生成中...'):
                    feedback_generator_output = feedback_chain.stream({
                        "rap": rap_result
                    })
                    feedback_result = ''.join(list(feedback_generator_output))
                st.write("### フィードバック")
                st.write(feedback_result)
                
                
            except Exception as e:
                st.error("処理中にエラーが発生しました。")
                st.error(traceback.format_exc())
                
# スタイル調整（オプション）
def style_adjustments():
    st.markdown(
    """
    <style>
    /* カスタムスタイル調整 */
    .st-emotion-cache-iiif1v { display: none !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )
    
if __name__ == '__main__':
    main()
    style_adjustments()