import functions_framework
import traceback
from flask import jsonify, request
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Constants
VALID_LANGUAGES = frozenset(["English", "Japanese"])


# プロンプトテンプレートの定義

# 既存のプロンプト定義は変更なし
PROMPT_JP = """
あなたはユーザーがメール返信を生成するのを助けるAI言語モデルです。メール会話の文脈を踏まえ、提供された入力に基づいて適切な構造の応答を作成します。応答は指定されたトーンと長さに合わせてください。
入力:
1. 送信者: メールを送る人（例: 上司、クライアントなど）
2. メールの件名: メールの件名（例: 会議のスケジュールについて）
3. メールのメッセージ: 送信者のメールの内容（例: 明日の会議の時間を調整したいのですが、午後は空いていますか？）
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
- 受信者のメールの内容:{message},
- あなたが言いたいこと:{reply},
---
"""

PROMPT_EN = """
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
- Content of the recipient's email:{message},
- What you want to say:{reply},
---
"""

def init_models(temperature=0):
    model = ChatOpenAI(temperature=temperature, model_name="gpt-4.1-2025-04-14")
    return model

def init_chain_jp():
    model = init_models()
    prompt = ChatPromptTemplate.from_messages([("user", PROMPT_JP)])
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    return chain

def init_chain_en():
    model = init_models()
    prompt = ChatPromptTemplate.from_messages([("user", PROMPT_EN)])
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    return chain

# ----------------------------------------------------------------
# ここから CORS 関連を複数ドメイン対応に修正
# ----------------------------------------------------------------

ALLOWED_ORIGINS = [
    "https://arigatoai.com",
    "https://www.arigatoai.com",
    "https://heysho.com",
    "https://www.heysho.com",
    # 開発環境で http://localhost:3000 等を許可したい場合は追加する
]

def create_cors_headers(request):
    """
    リクエストの Origin をチェックし、ALLOWED_ORIGINS に含まれている場合のみ許可する。
    含まれていなければ 'null' や '*' を返すなどの対応を考えられる。
    """
    origin = request.headers.get('Origin', '')
    if origin in ALLOWED_ORIGINS:
        allow_origin = origin
    else:
        # すべて許可したいなら "*"
        # 許可しない場合は "null" にするなどで対処
        allow_origin = "null"

    return {
        'Access-Control-Allow-Origin': allow_origin,
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '600',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }

def check_referer(request):
    referer = request.headers.get('Referer', '')
    # リファラで許可するドメイン一覧にも heysho.com を追加
    allowed_domains = [
        'arigatoai.com',
        'www.arigatoai.com',
        'heysho.com',
        'www.heysho.com'
    ]
    return any(domain in referer for domain in allowed_domains)

@functions_framework.http
def main(request):
    # Referer チェック
    if not check_referer(request):
        return (
            jsonify({'error': 'Unauthorized access'}),
            403,
            create_cors_headers(request)
        )

    # CORS プリフライト (ブラウザが OPTIONS メソッドで問い合わせてくる場合)
    if request.method == 'OPTIONS':
        headers = create_cors_headers(request)
        return ('', 204, headers)

    try:
        # リクエストの処理
        request_json = request.get_json(silent=True)
        request_args = request.args

        # message, reply, language の取得 (POST/GET どちらでも可)
        if request_json and 'message' in request_json:
            message = request_json['message']
            reply = request_json.get('reply')
            language = request_json.get('language')
        elif request_args and 'message' in request_args:
            message = request_args['message']
            reply = request_args.get('reply')
            language = request_args.get('language')
        else:
            return (
                jsonify({'error': 'No text provided'}),
                400,
                create_cors_headers(request)
            )

        # Validate language selection
        if language not in VALID_LANGUAGES:
            return (
                jsonify({'error': 'Invalid language selection'}),
                400,
                create_cors_headers(request)
            )

        # チェーンの初期化
        if language == "Japanese":
            chain = init_chain_jp()
        else:
            chain = init_chain_en()

        # 実行 (ストリーミングで受け取り、文字列にまとめる)
        output = chain.stream({
            "message": message,
            "reply": reply
        })
        result = ''.join(list(output))

        # 結果を JSON でレスポンス
        response_data = {
            'result': result,
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
