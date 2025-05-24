import traceback
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# プロンプトテンプレートの定義
###### Use dotenv if available ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################
# Prompt Template
PROMPT_1_JP = """
以下の話から、面白いエピソードトークを話ししてください。
オチとして、極端な比喩表現で笑かせてください。
- 行ったこと: {activity}

- 比喩表現の例：オリンピックが延期されると聞いた時は、残念すぎて膝から崩れ落ちて床を突き抜けて下の階の人と挨拶しました。
"""

# # 2. フィードバックプロンプト
# PROMPT_2_JP = """
# 以下の面白いエピソードトークを、わかりやすく、もっと面白くしてください。
# - 面白い比喩表現：{result_1}
# - 比喩表現の例：オリンピックが延期されると聞いた時は、残念すぎて膝から崩れ落ちて床を突き抜けて下の階の人と挨拶しました。
# """

def init_page():
    st.set_page_config(
        page_title="おもしろい例え話生成AIエージェント",
        page_icon="🎶"
    )
    st.header("おもしろい例え話生成AIエージェント 🎶")


def init_models(temperature=1):
    # model_1 = ChatOpenAI(temperature=temperature, model_name="gpt-4o")
    # Alternatively, if using other models:
    model_1 = ChatAnthropic(temperature=temperature, model_name="claude-3-5-sonnet-20241022")
    # model_1 = ChatGoogleGenerativeAI(temperature=temperature, model="gemini-exp-1121")

    # model_1 = ChatGoogleGenerativeAI(temperature=temperature, model="model-name")
    return model_1

def init_chain():
    model_1 = init_models()

    prompt_1 = ChatPromptTemplate.from_messages([("user", PROMPT_1_JP),])
    # prompt_2 = ChatPromptTemplate.from_messages([("user", PROMPT_2_JP),])
    # prompt_3 = ChatPromptTemplate.from_messages([("user", PROMPT_3_JP),])
    
    output_parser = StrOutputParser()
    
    # チェーンの構成
    chain_1 = prompt_1 | model_1 | output_parser
    # chain_2 = prompt_2| model_2 | output_parser
    
    return chain_1

def main():
    init_page()
    chain_1 = init_chain()
    if chain_1:
        activity = st.text_input("行ったこと", key="topic")
        # feeling = st.text_input("感想", key="feeling")

        if st.button("生成する"):
            try:
                # ステップ1: ラップ生成
                with st.spinner('生成中...'):
                    output_1 = chain_1.invoke({
                        "activity": activity,
                        # "feeling": feeling,
                    })
                    result = ''.join(list(output_1))
                st.write(result)
                
                # # ステップ2: フィードバック生成
                # with st.spinner('生成中...'):
                #     output_2 = chain_2.stream({
                #         "activity": activity,
                #         "feeling": feeling,
                #         "result_1": result_1,
                #     })
                #     result_2 = ''.join(list(output_2))
                # st.write("### 校正内容")
                # st.write(result_2)
                
                
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