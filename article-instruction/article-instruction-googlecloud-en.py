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

# ペルソナを表すデータモデル
class Persona(BaseModel):
    name: str = Field(..., description="読者ペルソナの名前")
    background: str = Field(..., description="ペルソナの持つ背景")

# ペルソナのリストを表すデータモデル
class Personas(BaseModel):
    personas: list[Persona] = Field(
        default_factory=list, description="読者ペルソナのリスト"
    )

# インタビュー内容を表すデータモデル
class Interview(BaseModel):
    persona: Persona = Field(..., description="インタビュー対象の読者ペルソナ")
    question: str = Field(..., description="インタビューでの質問")
    answer: str = Field(..., description="インタビューでの回答")

# インタビュー結果のリストを表すデータモデル
class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list, description="インタビュー結果のリスト"
    )

# 評価の結果を表すデータモデル
class EvaluationResult(BaseModel):
    reason: str = Field(..., description="判断の理由")
    is_sufficient: bool = Field(..., description="情報が十分かどうか")

# 要件定義生成AIエージェントのステート
class InterviewState(BaseModel):
    user_request: str = Field(..., description="ユーザーからのブログ記事のリクエスト")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list, description="生成された読者ペルソナのリスト"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list, description="実施されたインタビューのリスト"
    )
    requirements_doc: str = Field(default="", description="生成された記事作成の指示書")
    iteration: int = Field(
        default=0, description="ペルソナ生成とインタビューの反復回数"
    )
    is_information_sufficient: bool = Field(
        default=False, description="情報が十分かどうか"
    )

# ペルソナを生成するクラス
class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 2):
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
                    "各読者ペルソナには名前と簡単な背景を含めてください。年齢、性別、職業、トピックに対する知識レベルにおいて多様性を確保してください。",
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
        questions = self._generate_questions(
            user_request=user_request, personas=personas
        )
        answers = self._generate_answers(personas=personas, questions=questions)
        interviews = self._create_interviews(
            personas=personas, questions=questions, answers=answers
        )
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
                    "以下の読者ペルソナが、ブログ記事のトピックに関して自身の悩みや課題を話すための、オープンな質問を一つ作成してください。\n\n"
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
                    "あなたは以下の読者ペルソナです。インタビュアーの質問に対して、あなたが抱えている悩みや課題を具体的に教えてください。\n\nペルソナ:  {persona_name} - {persona_background}",
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
            Interview(persona=persona, question=question, answer=answer)
            for persona, question, answer in zip(personas, questions, answers)
        ]

# 情報の十分性を評価するクラス
class InformationEvaluator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(EvaluationResult)

    def run(self, user_request: str, interviews: list[Interview]) -> EvaluationResult:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは包括的な指示書を作成するための情報の十分性を評価する専門家です。",
                ),
                (
                    "human",
                    "以下のブログの記事内容とインタビュー結果に基づいて、包括的な要件文書を作成するのに十分な情報が集まったかどうかを判断してください。\n\n"
                    "トピック: {user_request}\n\n"
                    "インタビュー結果:\n{interview_results}",
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"ペルソナ: {i.persona.name} - {i.persona.background}\n"
                    f"質問: {i.question}\n回答: {i.answer}\n"
                    for i in interviews
                ),
            }
        )

# 要件定義書を生成するクラス
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
                    "以下のブログ記事のトピックと複数の読者ペルソナからのインタビュー結果に基づいて、記事作成の指示書を作成してください。\n\n"
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
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"ペルソナ: {i.persona.name} - {i.persona.background}\n"
                    f"質問: {i.question}\n回答: {i.answer}\n"
                    for i in interviews
                ),
            }
        )

# 要件定義書生成AIエージェントのクラス
class DocumentationAgent:
    def __init__(self, llm: ChatOpenAI, k: Optional[int] = None):
        self.persona_generator = PersonaGenerator(llm=llm, k=k)
        self.interview_conductor = InterviewConductor(llm=llm)
        self.information_evaluator = InformationEvaluator(llm=llm)
        self.requirements_generator = RequirementsDocumentGenerator(llm=llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(InterviewState)
        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("evaluate_information", self._evaluate_information)
        workflow.add_node("generate_requirements", self._generate_requirements)
        workflow.set_entry_point("generate_personas")
        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "evaluate_information")
        workflow.add_conditional_edges(
            "evaluate_information",
            lambda state: not state.is_information_sufficient and state.iteration < 5,
            {True: "generate_personas", False: "generate_requirements"},
        )
        workflow.add_edge("generate_requirements", END)
        return workflow.compile()

    def _generate_personas(self, state: InterviewState) -> dict[str, Any]:
        new_personas: Personas = self.persona_generator.run(state.user_request)
        return {
            "personas": new_personas.personas,
            "iteration": state.iteration + 1,
        }

    def _conduct_interviews(self, state: InterviewState) -> dict[str, Any]:
        new_interviews: InterviewResult = self.interview_conductor.run(
            state.user_request, state.personas[-5:]
        )
        return {"interviews": new_interviews.interviews}

    def _evaluate_information(self, state: InterviewState) -> dict[str, Any]:
        evaluation_result: EvaluationResult = self.information_evaluator.run(
            state.user_request, state.interviews
        )
        return {
            "is_information_sufficient": evaluation_result.is_sufficient,
            "evaluation_reason": evaluation_result.reason,
        }

    def _generate_requirements(self, state: InterviewState) -> dict[str, Any]:
        requirements_doc: str = self.requirements_generator.run(
            state.user_request, state.interviews
        )
        return {"requirements_doc": requirements_doc}

    def run(self, user_request: str) -> str:
        initial_state = InterviewState(user_request=user_request)
        final_state = self.graph.invoke(initial_state)
        return final_state["requirements_doc"]

# CORSヘッダーの作成
def create_cors_headers():
    return {
        'Access-Control-Allow-Origin': 'https://arigatoai.com',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '600',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }

# リファラーのチェック（必要に応じて）
def check_referer(request):
    referer = request.headers.get('Referer', '')
    allowed_domains = ['arigatoai.com', 'www.arigatoai.com']
    return any(domain in referer for domain in allowed_domains)

@functions_framework.http
def main(request):
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

        # user_requestの取得
        if request_json and 'user_request' in request_json:
            user_request = request_json['user_request']
            k = int(request_json.get('k', 3))
        elif request_args and 'user_request' in request_args:
            user_request = request_args['user_request']
            k = int(request_args.get('k', 3))
        else:
            return (jsonify({'error': 'user_requestが提供されていません'}), 400, create_cors_headers())

        # ChatOpenAIモデルを初期化
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)

        # 要件定義書生成AIエージェントを初期化
        agent = DocumentationAgent(llm=llm, k=k)

        # エージェントを実行して最終的な出力を取得
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