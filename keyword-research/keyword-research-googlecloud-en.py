import functions_framework
import traceback
from flask import jsonify, request
import operator
from typing import Annotated, Any, Optional
import os

# langchain-core, langchain-openai, langgraph 等は requirements.txt で指定してください
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

########################################
# データモデル
########################################

# ペルソナを表すデータモデル（検索者の特徴・背景）
class Persona(BaseModel):
    name: str = Field(..., description="検索ユーザーの名前")
    background: str = Field(..., description="検索ユーザーが抱える悩みや状況など")

# ペルソナのリストを表すデータモデル
class Personas(BaseModel):
    personas: list[Persona] = Field(
        default_factory=list,
        description="想定される検索ユーザーのリスト"
    )

# インタビュー（質問/回答）を表すデータモデル
class Interview(BaseModel):
    persona: Persona = Field(..., description="インタビュー対象のペルソナ")
    question: str = Field(..., description="インタビュアーからの質問（どんなキーワードで検索するかなど）")
    answer: str = Field(..., description="ペルソナの回答（実際に検索するとしたらどんなキーワードを使うか）")

# インタビュー結果のリストを表すデータモデル
class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list,
        description="すべてのインタビュー結果（ペルソナごとのキーワード候補など）"
    )

########################################
# SEOキーワード選定 エージェント用 State
########################################
class SEOInterviewState(BaseModel):
    user_request: str = Field(..., description="SEOキーワードを考えたいサービス・製品の情報")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list,
        description="生成された検索ユーザーのリスト"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list,
        description="インタビュー結果のリスト"
    )
    keyword_doc: str = Field(
        default="",
        description="最終的に生成されるSEOキーワード候補一覧"
    )

########################################
# PersonaGenerator
# (1) どんな検索者（ペルソナ）が想定されるか
########################################
class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 3):
        self.llm = llm.with_structured_output(Personas)  # 出力をPersonasモデルにパース
        self.k = k

    def run(self, service_desc: str) -> Personas:
        """
        service_desc を元に、多様な検索者像を k 人分生成
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはSEOのターゲットユーザー像を想定する専門家です。"
                ),
                (
                    "human",
                    f"以下のサービス(または製品)に興味を持ちそうな検索ユーザーを、{self.k}人のペルソナとして生成してください。\n\n"
                    "【サービス概要】\n{service_desc}\n\n"
                    "各ペルソナには：\n"
                    " - 名前\n"
                    " - 年齢・職業などの背景\n"
                    " - サービスを探す理由や悩み\n"
                    "などを具体的に書いてください。\n"
                    "出力は日本語でお願いします。",
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({"service_desc": service_desc})

########################################
# InterviewConductor
# (2) ペルソナごとに「どんなキーワードで検索するか」を尋ね、回答を得る
########################################
class InterviewConductor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, service_desc: str, personas: list[Persona]) -> InterviewResult:
        questions = self._generate_questions(service_desc, personas)
        answers = self._generate_answers(personas, questions)
        interviews = self._create_interviews(personas, questions, answers)
        return InterviewResult(interviews=interviews)

    def _generate_questions(self, service_desc: str, personas: list[Persona]) -> list[str]:
        """
        ペルソナに対して、「どんなキーワードで検索するか」を聞き出す質問を生成
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはSEOコンサルタントであり、検索ユーザーの生の声（どう検索するか）を引き出すインタビュアーです。"
                ),
                (
                    "human",
                    "以下のペルソナに対して、『あなたならどんなキーワードで検索エンジンを利用して、このサービスを探しそうですか？』と質問するための文面を1つ作成してください。\n\n"
                    "【サービス概要】\n{service_desc}\n"
                    "【ペルソナ】\n名前：{persona_name}\n背景：{persona_background}\n\n"
                    "質問はシンプルかつオープンエンドな形にしてください。"
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()

        queries = [
            {
                "service_desc": service_desc,
                "persona_name": p.name,
                "persona_background": p.background,
            }
            for p in personas
        ]
        return chain.batch(queries)

    def _generate_answers(self, personas: list[Persona], questions: list[str]) -> list[str]:
        """
        各ペルソナに対し、「こういうキーワードで検索しそう」と回答を作らせる
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは以下のペルソナ（検索ユーザー）です。インタビュアーに対し、実際に検索するとしたら使いそうなキーワードやフレーズを具体的に挙げてください。"
                ),
                (
                    "human",
                    "【ペルソナ情報】\n名前: {persona_name}\n背景: {persona_background}\n\n質問: {question}"
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()

        queries = [
            {
                "persona_name": persona.name,
                "persona_background": persona.background,
                "question": question,
            }
            for persona, question in zip(personas, questions)
        ]
        return chain.batch(queries)

    def _create_interviews(
        self, personas: list[Persona], questions: list[str], answers: list[str]
    ) -> list[Interview]:
        return [
            Interview(persona=persona, question=q, answer=a)
            for persona, q, a in zip(personas, questions, answers)
        ]

########################################
# KeywordDocumentGenerator
# (3) インタビュー結果をまとめ、SEOキーワード候補リストを作成
########################################
class KeywordDocumentGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, service_desc: str, interviews: list[Interview]) -> str:
        """
        ペルソナごとの回答をもとに、最終的なSEOキーワード候補リストを生成する
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはSEOキーワード選定の専門家です。"
                ),
                (
                    "human",
                    "以下の【サービス概要】と【インタビュー結果】から、ユーザーが検索しそうなキーワード候補一覧をまとめてください。\n"
                    "できるだけ多様な検索意図をカバーするようにキーワード案をリストアップし、日本語で出力してください。\n\n"
                    "【サービス概要】\n{service_desc}\n\n"
                    "【インタビュー結果】\n{interview_results}\n"
                    "それぞれのペルソナが挙げたキーワードを整理・分析し、\n"
                    "最終的な『SEOキーワード候補一覧』として提案してください。\n\n"
                    "キーワードに対する検索意図など、簡単なコメントもあると尚良いです。\n\n"
                    "出力は日本語でお願いします。"
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()

        interview_text = ""
        for i in interviews:
            interview_text += (
                f"▼ペルソナ: {i.persona.name} - {i.persona.background}\n"
                f" 質問: {i.question}\n"
                f" 回答: {i.answer}\n\n"
            )

        return chain.invoke({
            "service_desc": service_desc,
            "interview_results": interview_text
        })

########################################
# SEOキーワード選定エージェント
########################################
class SEOKeywordAgent:
    def __init__(self, llm: ChatOpenAI, k: Optional[int] = 3):
        self.persona_generator = PersonaGenerator(llm=llm, k=k)
        self.interview_conductor = InterviewConductor(llm=llm)
        self.keyword_generator = KeywordDocumentGenerator(llm=llm)

        # StateGraphの組み立て
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """
        1) ペルソナ生成 → 2) インタビュー実施 → 3) キーワード候補リスト生成 → END
        """
        workflow = StateGraph(SEOInterviewState)

        # ノード登録
        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("generate_keywords", self._generate_keywords)

        # エントリーポイント設定
        workflow.set_entry_point("generate_personas")

        # 遷移定義
        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "generate_keywords")
        workflow.add_edge("generate_keywords", END)

        # コンパイル
        return workflow.compile()

    def _generate_personas(self, state: SEOInterviewState) -> dict[str, Any]:
        # ペルソナを生成
        new_personas: Personas = self.persona_generator.run(state.user_request)
        return {
            "personas": state.personas + new_personas.personas
        }

    def _conduct_interviews(self, state: SEOInterviewState) -> dict[str, Any]:
        # 生成された最後の5人（あるいは全員）を対象にインタビュー
        new_personas = state.personas[-5:]
        interviews_result: InterviewResult = self.interview_conductor.run(
            state.user_request, new_personas
        )
        return {
            "interviews": state.interviews + interviews_result.interviews
        }

    def _generate_keywords(self, state: SEOInterviewState) -> dict[str, Any]:
        # インタビュー結果を元にSEOキーワード候補リストを作成
        keyword_doc: str = self.keyword_generator.run(
            state.user_request, state.interviews
        )
        return {"keyword_doc": keyword_doc}

    def run(self, service_desc: str) -> str:
        """
        メイン実行関数：ユーザーが提示するサービス内容を元に
        最終的な「SEOキーワード候補リスト」を返す
        """
        initial_state = SEOInterviewState(user_request=service_desc)
        final_state = self.graph.invoke(initial_state)
        return final_state["keyword_doc"]


########################################
# CORS & Referer の設定を複数ドメイン対応に修正
########################################

# arigatoai.com / heysho.com を両方許可したい場合のホワイトリスト
ALLOWED_ORIGINS = [
    "https://arigatoai.com",
    "https://www.arigatoai.com",
    "https://heysho.com",
    "https://www.heysho.com"
]

def create_cors_headers(request):
    """
    リクエストの Origin をチェックし、ALLOWED_ORIGINS に含まれていれば許可。
    そうでなければ 'null' や '*' にするなどの対応を行う。
    """
    origin = request.headers.get('Origin', '')
    if origin in ALLOWED_ORIGINS:
        allow_origin = origin
    else:
        # すべて許可したい場合は "*"
        # 特定だけ許可したい場合は "null" にする等
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
    リファラが arigatoai.com または heysho.com を含むかをチェック
    """
    referer = request.headers.get('Referer', '')
    allowed_domains = [
        'arigatoai.com',
        'www.arigatoai.com',
        'heysho.com',
        'www.heysho.com'
    ]
    return any(domain in referer for domain in allowed_domains)

########################################
# Cloud Run Functionsのエントリーポイント
########################################
@functions_framework.http
def main(request):
    # リファラーチェック
    if not check_referer(request):
        return (
            jsonify({'error': 'Unauthorized access'}),
            403,
            create_cors_headers(request)
        )

    # CORSプリフライト
    if request.method == 'OPTIONS':
        return ('', 204, create_cors_headers(request))

    try:
        # JSONボディまたはクエリパラメータから user_request (サービス概要) と k を取得
        request_json = request.get_json(silent=True)
        request_args = request.args

        if request_json:
            service_desc = request_json.get("service_desc", "")
            k = int(request_json.get("k", 3))
        elif request_args:
            service_desc = request_args.get("service_desc", "")
            k = int(request_args.get("k", 3))
        else:
            return (
                jsonify({'error': 'パラメータが提供されていません'}),
                400,
                create_cors_headers(request)
            )

        if not service_desc:
            return (
                jsonify({'error': 'service_desc が空です'}),
                400,
                create_cors_headers(request)
            )

        # ChatOpenAIモデルを初期化
        llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0.0)

        # エージェントを初期化
        agent = SEOKeywordAgent(llm=llm, k=k)

        # エージェント実行
        final_keywords = agent.run(service_desc=service_desc)

        response_data = {
            'result': final_keywords,
            'status': 'success'
        }
        return (
            jsonify(response_data),
            200,
            create_cors_headers(request)
        )

    except ValueError as ve:
        error_response = {
            'error': '無効な入力データ',
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
            'error': '内部サーバーエラー',
            'details': str(e),
            'traceback': traceback.format_exc(),
            'status': 'error'
        }
        return (
            jsonify(error_response),
            500,
            create_cors_headers(request)
        )
