import functions_framework
import traceback
from flask import jsonify, request
import operator
from typing import Annotated, Any, Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
import os

# -----------------------------------------------------------------------------
# データモデル定義
# -----------------------------------------------------------------------------

# ペルソナを表すデータモデル
class Persona(BaseModel):
    name: str = Field(..., description="ペルソナの名前")
    background: str = Field(..., description="ペルソナの持つ背景")

# ペルソナのリストを表すデータモデル
class Personas(BaseModel):
    personas: list[Persona] = Field(
        default_factory=list, description="ペルソナのリスト"
    )

# インタビュー内容を表すデータモデル
class Interview(BaseModel):
    persona: Persona = Field(..., description="インタビュー対象のペルソナ")
    question: str = Field(..., description="インタビューでの質問")
    answer: str = Field(..., description="インタビューでの回答")

# インタビュー結果のリストを表すデータモデル
class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list, description="インタビュー結果のリスト"
    )

# 要件定義生成AIエージェントのステート
class InterviewState(BaseModel):
    user_request: str = Field(..., description="ユーザーからのリクエスト（ブログ記事のトピック）")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list, description="生成されたペルソナのリスト"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list, description="実施されたインタビューのリスト"
    )
    requirements_doc: str = Field(default="", description="生成されたブログ記事作成の指示書")
    iteration: int = Field(
        default=0, description="ペルソナ生成とインタビューの反復回数"
    )
    # 情報評価ステップ削除に伴い、このフラグ等は削除

# -----------------------------------------------------------------------------
# 各ステップのクラス定義
# -----------------------------------------------------------------------------

# ペルソナを生成するクラス
class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 5):
        # llm.with_structured_output(...) によって、LLMの出力を pydanticモデルにパース
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはブログ記事のターゲットユーザーへのインタビュー用の多様なペルソナを作成する専門家です。",
                ),
                (
                    "human",
                    f"以下のブログ記事のトピックに関するインタビュー用に、{self.k}人の多様なペルソナを生成してください。\n\n"
                    "トピック: {user_request}\n\n"
                    "各読者ペルソナには名前と簡単な背景を含めてください。"
                    "年齢、性別、職業、トピックに対する知識レベルにおいて多様性を確保してください。",
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({"user_request": user_request})

# インタビューを実施するクラス
class InterviewConductor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, personas: list[Persona]) -> InterviewResult:
        # 1. 質問を生成
        questions = self._generate_questions(
            user_request=user_request, personas=personas
        )
        # 2. 回答を生成
        answers = self._generate_answers(personas=personas, questions=questions)
        # 3. 組み合わせからインタビューリストを作成
        interviews = self._create_interviews(personas, questions, answers)
        return InterviewResult(interviews=interviews)

    def _generate_questions(
        self, user_request: str, personas: list[Persona]
    ) -> list[str]:
        question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはインタビュアーです。ペルソナの悩みや課題を引き出すための質問を作成します。",
                ),
                (
                    "human",
                    "以下の読者ペルソナが、ブログ記事のトピックに関して自身の悩みや課題を話すための、"
                    "オープンな質問を一つ作成してください。\n\n"
                    "トピック: {user_request}\n"
                    "読者ペルソナ: {persona_name} - {persona_background}\n\n"
                    "質問はシンプルで、このペルソナが自分の悩みを率直に話せるようにしてください。",
                ),
            ]
        )
        question_chain = question_prompt | self.llm | StrOutputParser()

        question_queries = [
            {
                "user_request": user_request,
                "persona_name": persona.name,
                "persona_background": persona.background,
            }
            for persona in personas
        ]
        return question_chain.batch(question_queries)

    def _generate_answers(
        self, personas: list[Persona], questions: list[str]
    ) -> list[str]:
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは以下の読者ペルソナです。インタビュアーの質問に対して、"
                    "あなたが抱えている悩みや課題を具体的に教えてください。\n\n"
                    "ペルソナ: {persona_name} - {persona_background}",
                ),
                ("human", "質問: {question}"),
            ]
        )
        answer_chain = answer_prompt | self.llm | StrOutputParser()

        answer_queries = [
            {
                "persona_name": persona.name,
                "persona_background": persona.background,
                "question": question,
            }
            for persona, question in zip(personas, questions)
        ]
        return answer_chain.batch(answer_queries)

    def _create_interviews(
        self, personas: list[Persona], questions: list[str], answers: list[str]
    ) -> list[Interview]:
        return [
            Interview(persona=persona, question=q, answer=a)
            for persona, q, a in zip(personas, questions, answers)
        ]

# ブログ記事指示書を生成するクラス
class RequirementsDocumentGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, interviews: list[Interview]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは収集した情報に基づいて記事作成の指示書を作成する専門家です。",
                ),
                (
                    "human",
                    "以下のブログ記事のトピックと複数の読者ペルソナからのインタビュー結果に基づいて、"
                    "記事作成の指示書を作成してください。\n\n"
                    "トピック: {user_request}\n\n"
                    "インタビュー結果:\n{interview_results}\n"
                    "記事作成の指示書には以下のセクションを含めてください:\n"
                    "1. 記事の目的\n"
                    "2. ターゲット読者\n"
                    "3. 読者の悩み\n"
                    "4. SEOのターゲットキーワードとトピック\n"
                    "5. 記事の構成案\n"
                    "6. 注意事項\n"
                    "出力は必ず日本語でお願いします。\n\n記事作成の指示書:",
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()

        # インタビュー結果をテキスト形式にまとめる
        interview_results_text = "\n".join(
            f"ペルソナ: {i.persona.name} - {i.persona.background}\n"
            f"質問: {i.question}\n回答: {i.answer}\n"
            for i in interviews
        )

        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": interview_results_text,
            }
        )

# -----------------------------------------------------------------------------
# エージェント本体
# -----------------------------------------------------------------------------
class DocumentationAgent:
    def __init__(self, llm: ChatOpenAI, k: Optional[int] = None):
        self.persona_generator = PersonaGenerator(llm=llm, k=k)
        self.interview_conductor = InterviewConductor(llm=llm)
        self.requirements_generator = RequirementsDocumentGenerator(llm=llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        # ステートグラフを作成（情報評価ステップは削除）
        workflow = StateGraph(InterviewState)

        # ノード追加
        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("generate_requirements", self._generate_requirements)

        # エントリーポイント
        workflow.set_entry_point("generate_personas")

        # 遷移設定
        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "generate_requirements")
        workflow.add_edge("generate_requirements", END)

        return workflow.compile()

    def _generate_personas(self, state: InterviewState) -> dict[str, Any]:
        new_personas: Personas = self.persona_generator.run(state.user_request)
        return {
            "personas": new_personas.personas,
            "iteration": state.iteration + 1,
        }

    def _conduct_interviews(self, state: InterviewState) -> dict[str, Any]:
        # ペルソナが多い場合は最後の5人のみに絞る（同様の挙動を踏襲）
        new_personas = state.personas[-5:]
        new_interviews: InterviewResult = self.interview_conductor.run(
            user_request=state.user_request,
            personas=new_personas
        )
        return {"interviews": new_interviews.interviews}

    def _generate_requirements(self, state: InterviewState) -> dict[str, Any]:
        requirements_doc: str = self.requirements_generator.run(
            user_request=state.user_request,
            interviews=state.interviews
        )
        return {"requirements_doc": requirements_doc}

    def run(self, user_request: str) -> str:
        # ステート初期化
        initial_state = InterviewState(user_request=user_request)
        # ステートマシン実行
        final_state = self.graph.invoke(initial_state)
        return final_state["requirements_doc"]

# -----------------------------------------------------------------------------
# CORS関連ヘルパー
# -----------------------------------------------------------------------------
def create_cors_headers():
    return {
        'Access-Control-Allow-Origin': 'https://heysho.com',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '600',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }

def check_referer(request):
    referer = request.headers.get('Referer', '')
    allowed_domains = ['heysho.com', 'www.heysho.com']
    return any(domain in referer for domain in allowed_domains)

# -----------------------------------------------------------------------------
# Cloud Run Functionsエントリーポイント
# -----------------------------------------------------------------------------
@functions_framework.http
def main(request):
    # リファラーのチェック（必要に応じて）
    if not check_referer(request):
        return (jsonify({'error': 'Unauthorized access'}), 403, create_cors_headers())

    # CORSのプリフライトチェック
    if request.method == 'OPTIONS':
        headers = create_cors_headers()
        return ('', 204, headers)

    try:
        # リクエストデータの取得
        request_json = request.get_json(silent=True)
        request_args = request.args

        # user_request と k の取得
        if request_json and 'user_request' in request_json:
            user_request = request_json['user_request']
            k = int(request_json.get('k', 3))
        elif request_args and 'user_request' in request_args:
            user_request = request_args['user_request']
            k = int(request_args.get('k', 3))
        else:
            return (
                jsonify({'error': 'user_requestが提供されていません'}),
                400,
                create_cors_headers()
            )

        # ChatOpenAIモデルを初期化
        # 必要に応じて model_name を変更してください
        llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0.0)

        # エージェントを初期化し、実行
        agent = DocumentationAgent(llm=llm, k=k)
        final_output = agent.run(user_request=user_request)

        response_data = {
            'result': final_output,
            'status': 'success'
        }
        return (jsonify(response_data), 200, create_cors_headers())

    except ValueError as ve:
        error_response = {
            'error': '無効な入力データ',
            'details': str(ve),
            'status': 'error'
        }
        return (jsonify(error_response), 400, create_cors_headers())

    except Exception as e:
        error_response = {
            'error': '内部サーバーエラー',
            'details': str(e),
            'traceback': traceback.format_exc(),
            'status': 'error'
        }
        return (jsonify(error_response), 500, create_cors_headers())
