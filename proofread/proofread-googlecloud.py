import functions_framework
import traceback
from flask import jsonify
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI

# Constants
VALID_LANGUAGES = frozenset(["English", "Japanese"])


# プロンプトテンプレートの定義

# 既存のプロンプト定義は変更なし
PROMPT_1_JP = """
次の文章を校正し、文法や語彙の正確さを確認してください。校正する際の注意点は以下の通りです:
- 文法上の誤りを修正し、正確な文章にしてください。
- 文章全体の一貫性を保ち、各文が自然な流れであるかを確認してください。
- 語彙の選択が適切で、表現が明確であるかをチェックし、必要に応じて改善してください。
- 希望内容に応じて調整してください。
文章: {text}
希望内容: {user_need}
"""

PROMPT_2_JP = """
次の文章を最終校正し、全体のトーンやニュアンスを自然で読みやすい形に仕上げてください。校正の際は以下の点に留意してください:
- 文章全体の流れやリズムが自然で、読み手にスムーズに伝わるように調整してください。
- 冗長な部分を整理し、簡潔でわかりやすい表現にしてください。
- 語彙や言い回しをターゲット読者に合ったものにし、文化的なニュアンスにも配慮してください。
- フィードバックを反映して改善ください。
- 希望内容を反映して最終調整を行ってください。
文章: {result_1}
希望内容: {user_need}
"""

# 1. First Proofread Prompt
PROMPT_1_EN = """
Please proofread the following text and check for grammatical and vocabulary accuracy. When proofreading, please pay attention to the following:
- Correct any grammatical errors to ensure the text is accurate.
- Maintain overall consistency and ensure each sentence flows naturally.
- Check that vocabulary choices are appropriate and expressions are clear, improving them if necessary.
- Adjust according to the desired content.
Text: {text}
Desired Content: {user_need}
"""

# 3. Improvement Prompt
PROMPT_2_EN = """
Please perform a final proofread of the following text, refining the overall tone and nuance to make it natural and easy to read. When proofreading, please pay attention to the following:
- Adjust the overall flow and rhythm of the text to ensure it is smooth and easily understood by the reader.
- Eliminate redundant parts and use concise and clear expressions.
- Choose vocabulary and phrasing that suit the target audience, considering cultural nuances.
- Reflect the feedback to make improvements.
- Make final adjustments to align with the desired content.
Text: {result_1}
Desired Content: {user_need}
"""

def init_models(temperature=0):
    # model_1 = ChatGoogleGenerativeAI(temperature=temperature, model="gemini-1.5-flash")
    model_1 = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=temperature)
    model_2 = ChatAnthropic(temperature=temperature,model_name="claude-3-7-sonnet-latest")
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
        # 許可しない場合やワイルドカードにするなどの対応をここで決める
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
    allowed_domains = ['arigatoai.com', 'www.arigatoai.com','heysho.com', 'www.heysho.com']
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
        if request_json and 'text' in request_json:
            text = request_json['text']
            user_need = request_json['user_need']
            language = request_json['language']

        elif request_args and 'text' in request_args:
            text = request_args['text']
            user_need = request_args.get('user_need')
            language = request_args.get('language')
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
            "text": text,
            "user_need":user_need
            })
        result_1 = ''.join(list(output_1))

        # ステップ2: 校正
        output_2 = chain_2.stream({
            "result_1": result_1,
            "user_need":user_need
            })
        result_2 = ''.join(list(output_2))

        response_data = {
            'result_3': result_2,
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
