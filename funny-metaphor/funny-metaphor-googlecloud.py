import functions_framework
import traceback
from flask import jsonify, request
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

VALID_LANGUAGES = frozenset(["English", "Japanese"])

# プロンプトテンプレートの定義
PROMPT_1_JP = """
# あなたの役割
あなたは落語家とスタンダップコメディアンが合体したAIストーリーテラーです。
聞き手を笑わせることが最優先。最後の１文で〈物理法則を無視するレベルの極端な比喩〉を必ず入れてください。

# 入力
- 行ったこと（activity）: {activity}

# 出力ルール
1. 3〜5文でコンパクトに。
2. 1〜2文目: 体験の概要をテンポよく描写。
3. 3〜4文目: ちょっとした伏線や気持ちの盛り上げ。
4. 5文目: “オチ”。現実離れした比喩で爆発的に誇張し、必ず笑いを誘う。

# 比喩表現の参考例（必要に応じて流用・改変可）
オリンピック延期と聞いた瞬間、膝から崩れ落ちて床を突き抜け下の階の人と目があって「はじめまして」と挨拶しました。

# さあ、面白エピソードをどうぞ！

"""

PROMPT_1_EN = """
# Your role
You are an AI storyteller who blends Japanese rakugo timing with stand-up comedy punchiness.
Your mission: make the listener laugh. End with ONE sentence containing an *absurd, physics-defying metaphor*.

# Input
- Activity: {activity}

# Output rules
1. Keep it to 3-5 sentences.
2. Sentences 1–2: set up the scene briskly.
3. Sentences 3–4: build tension or emotion.
4. Sentence 5: the “punchline” — an over-the-top metaphor that detonates the humor.

# Metaphor cheat-sheet (feel free to adapt or remix)
1. When I heard the Olympics were postponed, I crashed through the floor and greeted my neighbors and said Nice to meet you.

# Now, spin your hilarious episode!
"""

def init_models(temperature=1):
    """
    Anthropic のモデルを初期化する関数
    """
    model_1 = ChatAnthropic(
        temperature=temperature,
        model_name="claude-3-7-sonnet-latest"
    )
    return model_1

def init_chain_jp():
    """
    日本語のプロンプトチェーンを初期化
    """
    model_1 = init_models()
    prompt_1 = ChatPromptTemplate.from_messages([
        ("user", PROMPT_1_JP)
    ])
    output_parser = StrOutputParser()
    chain_1 = prompt_1 | model_1 | output_parser
    return chain_1

def init_chain_en():
    """
    英語のプロンプトチェーンを初期化
    """
    model_1 = init_models()
    prompt_1 = ChatPromptTemplate.from_messages([
        ("user", PROMPT_1_EN)
    ])
    output_parser = StrOutputParser()
    chain_1 = prompt_1 | model_1 | output_parser
    return chain_1


# このリストに含まれる Origin だけを許可する例
# 必要があれば開発用に "http://localhost:3000" 等を追加してください
ALLOWED_ORIGINS = [
    'https://arigatoai.com',
    'https://www.arigatoai.com',
    'https://heysho.com',
    'https://www.heysho.com'
]

def create_cors_headers(request):
    """
    リクエストの Origin を確認して、ホワイトリストに含まれていればそれを許可。
    含まれていなければ 'null' や '*' を返す。
    """
    origin = request.headers.get('Origin', '')
    if origin in ALLOWED_ORIGINS:
        allow_origin = origin
    else:
        # 許可したくない場合は 'null'、あるいは全部許可したいなら '*' などにする
        allow_origin = "null"

    return {
        'Access-Control-Allow-Origin': allow_origin,
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '600',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }

def check_referer(request):
    """
    Referer ヘッダーを簡易チェックする関数
    ここに含まれるドメインのいずれかが referer にマッチすれば True を返す
    """
    referer = request.headers.get('Referer', '')
    allowed_domains = [
        'arigatoai.com',
        'www.arigatoai.com',
        'heysho.com',
        'www.heysho.com'
    ]
    return any(domain in referer for domain in allowed_domains)

@functions_framework.http
def main(request):
    """
    メインのエンドポイント
    """

    # Referer での簡易的なドメイン制限
    if not check_referer(request):
        # 403 を返すが、CORS ヘッダーをつけるために request を渡す
        return (
            jsonify({'error': 'Unauthorized access'}),
            403,
            create_cors_headers(request)
        )

    # CORSプリフライト対応 (ブラウザが OPTIONS を投げてきたときに応答)
    if request.method == 'OPTIONS':
        headers = create_cors_headers(request)
        return ('', 204, headers)

    try:
        # リクエストボディ or クエリパラメータの解析
        request_json = request.get_json(silent=True)
        request_args = request.args

        # POST (JSON) または GET (Query) の両方に対応
        if request_json and 'activity' in request_json:
            activity = request_json['activity']
            language = request_json['language']
        elif request_args and 'activity' in request_args:
            activity = request_args['activity']
            language = request_args['language']
        else:
            return (
                jsonify({'error': 'No text provided'}),
                400,
                create_cors_headers(request)
            )

        # 言語選択のバリデーション
        if language not in VALID_LANGUAGES:
            return (
                jsonify({'error': 'Invalid language selection'}),
                400,
                create_cors_headers(request)
            )

        # チェーンの初期化 & 面白い比喩表現の生成
        if language == "Japanese":
            chain_1 = init_chain_jp()
        else:
            chain_1 = init_chain_en()

        # ストリーミング出力をまとめて文字列に結合
        output_1 = chain_1.stream({"activity": activity})
        result_1 = ''.join(list(output_1))

        # 結果を JSON で返す
        response_data = {
            'result': result_1,
            'status': 'success'
        }
        return (
            jsonify(response_data),
            200,
            create_cors_headers(request)
        )

    except ValueError as ve:
        error_response = {
            'error': 'Invalid input data',
            'details': str(ve),
            'status': 'error'
        }
        return (
            jsonify(error_response),
            400,
            create_cors_headers(request)
        )

    except Exception as e:
        error_response = {
            'error': 'Internal server error',
            'details': str(e),
            'traceback': traceback.format_exc(),
            'status': 'error'
        }
        return (
            jsonify(error_response),
            500,
            create_cors_headers(request)
        )
