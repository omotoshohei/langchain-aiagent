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

# プロンプトテンプレートの定義

# 1. ラップ生成プロンプト
FIRST_PROOFREAD_PROMPT = """
次の文章を校正し、文法や語彙の正確さを確認してください。校正する際の注意点は以下の通りです:
- 文法上の誤りを修正し、正確な文章にしてください。
- 文章全体の一貫性を保ち、各文が自然な流れであるかを確認してください。
- 語彙の選択が適切で、表現が明確であるかをチェックし、必要に応じて改善してください。
- 希望内容に応じて調整してください。
文章: {text}
希望内容: {user_need}
"""

# 2. フィードバックプロンプト
FEEDBACK_PROMPT = """
次の校正結果に対してフィードバックを行い、さらに改善点を提案してください。フィードバックする際の注意点は以下の通りです:
- 校正された文章に残っている文法、語彙、表現の問題点を指摘してください。
- 改善が必要な箇所に対して具体的な修正案を提案してください。
- 全体の読みやすさやトーンが適切であるかも考慮してください。
- 希望内容に沿ったフィードバックを行ってください。
- 例文は提示しないでください。
希望内容: {user_need}
校正結果: {proofread_result}
"""

# 3. 改善プロンプト
SECOND_PROOFREAD_PROMPT = """
次の文章を最終校正し、全体のトーンやニュアンスを自然で読みやすい形に仕上げてください。校正の際は以下の点に留意してください:
- 文章全体の流れやリズムが自然で、読み手にスムーズに伝わるように調整してください。
- 冗長な部分を整理し、簡潔でわかりやすい表現にしてください。
- 語彙や言い回しをターゲット読者に合ったものにし、文化的なニュアンスにも配慮してください。
- フィードバックを反映して改善ください。
- 希望内容を反映して最終調整を行ってください。
文章: {proofread_result}
希望内容: {user_need}
フィードバック: {feedback_result}
"""

def init_page():
    st.set_page_config(
        page_title="文章校正AIエージェント",
        page_icon="🎶"
    )
    st.header("文章校正AIエージェント 🎶")

# Function to initialize models
def init_models():
    first_proofread_model = ChatAnthropic(temperature=0, model_name="claude-3-5-haiku-20241022")
    feedback_model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    second_proofread_model = ChatAnthropic(temperature=1, model_name="claude-3-5-sonnet-20241022")
    return first_proofread_model, feedback_model, second_proofread_model

def init_chain():
    first_proofread_model, feedback_model, second_proofread_model = init_models()
    
    # 各プロンプトと出力パーサーの設定
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
    
    # チェーンの構成
    first_proofread_chain = first_proofread_prompt | first_proofread_model | output_parser
    feedback_chain = feedback_prompt | feedback_model | output_parser
    second_proofread_chain = second_proofread_prompt | second_proofread_model | output_parser
    
    return first_proofread_chain, feedback_chain, second_proofread_chain

def main():
    init_page()
    first_proofread_chain, feedback_chain, second_proofread_chain = init_chain()
    if first_proofread_chain and feedback_chain and second_proofread_chain:
        text = st.text_area("校正するテキスト", key="text")
        user_need = st.text_input("希望する内容（オプショナル）", key="user_need")
        if st.button("文章を校正する"):
            try:
                # ステップ1: ラップ生成
                with st.spinner('文章を校正中...'):
                    first_proofread_output = first_proofread_chain.stream({
                        "text": text,
                        "user_need": user_need,
                    })
                    first_proofread_result = ''.join(list(first_proofread_output))
                st.write("### 1回目の校正")
                st.write(first_proofread_result)
                
                # ステップ2: フィードバック生成
                with st.spinner('フィードバックを生成中...'):
                    feedback_output = feedback_chain.stream({
                        "proofread_result": first_proofread_result,
                        "user_need": user_need,
                    })
                    feedback_result = ''.join(list(feedback_output))
                st.write("### フィードバック")
                st.write(feedback_result)
                
                # ステップ3: ラップの改善
                with st.spinner('文章を改善中...'):
                    final_text_output = second_proofread_chain.stream({
                        "proofread_result": first_proofread_result,
                        "user_need": user_need,
                        "feedback_result": feedback_result,
                    })
                    final_text = ''.join(list(final_text_output))
                st.write("### 改善された文章")
                st.write(final_text)
                
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