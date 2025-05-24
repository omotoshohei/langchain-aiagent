import functions_framework
import traceback
from flask import jsonify
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Constants
VALID_LANGUAGES = frozenset(["English", "Japanese"])


# プロンプトテンプレートの定義

# 既存のプロンプト定義は変更なし
PROMPT_1_JP = """
下の内容をテーマにした、8ビートの4小節16行の日本語ラップを書いてください。
韻を多く踏んでください。
- トピック: {topic}
- 言いたいこと: {message}
"""

PROMPT_2_JP = """
8ビートの4小節16行のラップになっていることを確認して、間違ってたら16行目以降を省略して。
四行ごとに[Verse 1][Verse 2][Verse 3][Verse 4]と見出しをつけて。
- ラップ：{result_1}
"""
# 1. First Proofread Prompt
PROMPT_1_EN = """
Write a 16-line rap (4 bars) based with 8 beat on the content below:
- Include as many rhymes as possible
- Strictly maintain 16-line
- Topic: {topic}
- What you want to say: {message}
"""

# 2. Feedback Prompt
PROMPT_2_EN = """
Check if the rap is exactly 16 lines (4 bars) with 8 beat, and if it's longer than 16 lines, omit any lines after line 17.
- Add headings [Verse 1], [Verse 2], [Verse 3], and [Verse 4] every four lines.
- Rap:{result_1}
"""

def init_models(temperature=0):
    model_1 = ChatOpenAI(temperature=temperature, model_name="gpt-4.1-2025-04-14")
    model_2 = ChatGoogleGenerativeAI(temperature=temperature, model="gemini-2.0-flash-lite")
    return model_1, model_2

def init_chain_jp():
    model_1, model_2 = init_models()
    prompt_1 = ChatPromptTemplate.from_messages([("user", PROMPT_1_JP)])
    prompt_2 = ChatPromptTemplate.from_messages([("user", PROMPT_2_JP)])
    output_parser = StrOutputParser()
    chain_1 = prompt_1 | model_1 | output_parser
    chain_2 = prompt_2 | model_2 | output_parser
    return chain_1, chain_2

def init_chain_en():
    model_1, model_2 = init_models()
    prompt_1 = ChatPromptTemplate.from_messages([("user", PROMPT_1_EN)])
    prompt_2 = ChatPromptTemplate.from_messages([("user", PROMPT_2_EN)])
    output_parser = StrOutputParser()
    chain_1 = prompt_1 | model_1 | output_parser
    chain_2 = prompt_2 | model_2 | output_parser
    return chain_1, chain_2


ALLOWED_ORIGINS = ['https://arigatoai.com', 'https://www.arigatoai.com','https://heysho.com', 'https://www.heysho.com']

def create_cors_headers(request):
    origin = request.headers.get('Origin', '')
    # リクエストの Origin がホワイトリストに含まれているかチェック
    if origin in ALLOWED_ORIGINS:
        allow_origin = origin
    else:
        allow_origin = "null"  # あるいは "*"

    return {
        'Access-Control-Allow-Origin': allow_origin,
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '600',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }

def check_referer(request):
    referer = request.headers.get('Referer', '')
    allowed_domains = [
        'arigatoai.com', 'www.arigatoai.com',
        'heysho.com', 'www.heysho.com'
    ]
    return any(domain in referer for domain in allowed_domains)

@functions_framework.http
def main(request):
    # リファラーチェック
    if not check_referer(request):
        return (jsonify({'error': 'Unauthorized access'}), 403, create_cors_headers(request))

    if request.method == 'OPTIONS':
        headers = create_cors_headers(request)
        return ('', 204, headers)

    try:
        # リクエストの処理
        request_json = request.get_json(silent=True)
        request_args = request.args
        # POSTとGETの両方に対応
        if request_json and 'message' in request_json:
            topic = request_json['topic']
            message = request_json['message']
            language = request_json['language']

        elif request_args and 'message' in request_args:
            topic = request_args['topic']
            message = request_args['message']
            language = request_args['language']
        else:
            return (jsonify({'error': 'No text provided'}), 400, create_cors_headers(request))
        # Validate language selection
        if language not in VALID_LANGUAGES:
            return (jsonify({'error': 'Invalid language selection'}), 400, create_cors_headers(request))

        # 校正処理の実行
        if language == "Japanese":
            chain_1, chain_2 = init_chain_jp()
        else:
            chain_1, chain_2 = init_chain_en()

        # ステップ1: 翻訳
        output_1 = chain_1.stream({
            "topic": topic,
            "message":message,
            })
        result_1 = ''.join(list(output_1))

        # ステップ2: 校正
        output_2 = chain_2.stream({
            "result_1": result_1,
            })
        final_result = ''.join(list(output_2))


        response_data = {
            'final_result': final_result,
            'status': 'success'
        }
        return (jsonify(response_data), 200, create_cors_headers(request))

    except ValueError as ve:
        error_response = {
            'error': 'Invalid input data',
            'details': str(ve),
            'status': 'error'
        }
        return (jsonify(error_response), 400, create_cors_headers(request))

    except Exception as e:
        error_response = {
            'error': 'Internal server error',
            'details': str(e),
            'traceback': traceback.format_exc(),
            'status': 'error'
        }
        return (jsonify(error_response), 500, create_cors_headers(request))
