import functions_framework
import traceback
from flask import jsonify, request
import operator
from typing import Annotated, Any, Optional

# -------------------------------
# langchain & pydantic imports
# -------------------------------
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

import os

# -----------------------------------------------------------------------------
# (Optional) Set environment variables here or rely on your Cloud Run’s config
# e.g.:
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
# os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGCHAIN_API_KEY"
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# CORS / Referer checking helpers (same as your reference)
# -----------------------------------------------------------------------------
ALLOWED_ORIGINS = [
    "https://arigatoai.com",
    "https://www.arigatoai.com",
    "https://heysho.com",
    "https://www.heysho.com",
    # Add localhost if needed during development
    # "http://localhost:3000",
]

def create_cors_headers(request):
    origin = request.headers.get('Origin', '')
    if origin in ALLOWED_ORIGINS:
        allow_origin = origin
    else:
        allow_origin = "null"  # or "*"
    return {
        'Access-Control-Allow-Origin': allow_origin,
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '600',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }

def check_referer(request):
    referer = request.headers.get('Referer', '')
    allowed_domains = ['arigatoai.com', 'www.arigatoai.com', 'heysho.com', 'www.heysho.com']
    return any(domain in referer for domain in allowed_domains)

# -----------------------------------------------------------------------------
# Data Models (from your Colab code)
# -----------------------------------------------------------------------------
class Persona(BaseModel):
    name: str = Field(..., description="ペルソナの名前")
    background: str = Field(..., description="ペルソナの背景情報")

class Personas(BaseModel):
    personas: list[Persona] = Field(
        default_factory=list, description="ペルソナのリスト"
    )

class Interview(BaseModel):
    persona: Persona = Field(..., description="インタビュー対象のペルソナ")
    question: str = Field(..., description="インタビューでの質問")
    answer: str = Field(..., description="インタビューでの回答")

class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list, description="インタビュー結果のリスト"
    )

class InterviewState(BaseModel):
    user_request: str = Field(..., description="ユーザーからのリクエスト")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list, description="生成されたペルソナのリスト"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list, description="実施されたインタビューのリスト"
    )
    seo_strategy_doc: str = Field(default="", description="生成されたSEO戦略ドキュメント")
    iteration: int = Field(
        default=0, description="ペルソナ生成とインタビューの反復回数"
    )

# -----------------------------------------------------------------------------
# Classes for each step (PersonaGenerator, InterviewConductor, etc.)
# -----------------------------------------------------------------------------
class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 5):
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        # --- SEO向けに変更したプロンプト ここから ---
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは依頼元が提供するサービスに関心を持ち得る潜在顧客のペルソナ作成に特化した専門家です。"
                    "それぞれのペルソナが具体的にどんな悩みを抱え、どんな情報を求めているかを明確に示してください。"
                ),
                (
                    "human",
                    (
                        f"以下の内容で、{self.k}人の多様なペルソナを生成してください。\n\n"
                        "【サービスの概要】\n{user_request}\n\n"
                        "各ペルソナには以下を含めてください:\n"
                        "- 名前\n"
                        "- 年齢、性別、職業\n"
                        "- どんな課題・悩みを抱えているか\n"
                        "- あなたのサービスを見つける可能性が高い情報源（検索エンジン、SNS、知人の紹介など）\n"
                        "- どんな情報（価格、口コミ、機能比較など）を特に重視するか\n"
                        "- どんなキーワードで検索しそうか（推定でOK）\n"
                        "具体的でリアルに想像しやすい設定をお願いします。"
                    ),
                ),
            ]
        )
        # --- SEO向けに変更したプロンプト ここまで ---
        chain = prompt | self.llm
        return chain.invoke({"user_request": user_request})

class InterviewConductor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, personas: list[Persona]) -> InterviewResult:
        questions = self._generate_questions(user_request, personas)
        answers = self._generate_answers(personas, questions)
        interviews = self._create_interviews(personas, questions, answers)
        return InterviewResult(interviews=interviews)

    def _generate_questions(self, user_request: str, personas: list[Persona]) -> list[str]:
        question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはインタビュアーです。潜在顧客の悩みや課題を深掘りし、"
                    "どのようにサービスを見つけ、選び、利用しようとしているのかを明らかにする質問を考えるプロです。"
                ),
                (
                    "human",
                    (
                        "以下の読者ペルソナが、あなたのサービスに対して抱えている悩みや、"
                        "検索やSNSを含む情報収集の実態を率直に話せるようなオープンな質問を作成してください。\n\n"
                        "【サービスの概要】\n{user_request}\n\n"
                        "【読者ペルソナ】\n{persona_name} - {persona_background}\n\n"
                        "質問のポイント:\n"
                        "- 現在の悩みや目的を具体的に引き出せる\n"
                        "- どこで情報収集しているかを明確にできる\n"
                        "- 何を重視して比較検討しているのかを知る\n"
                        "シンプルだが、回答者が深く考えられるようにしてください。"
                    ),
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

    def _generate_answers(self, personas: list[Persona], questions: list[str]) -> list[str]:
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは以下のペルソナです。サービスに関して抱えている具体的な悩み、"
                    "どのように情報を探すのか、何を基準に選ぶのかなどを正直に答えてください。"
                ),
                (
                    "human",
                    (
                        "ペルソナ: {persona_name} - {persona_background}\n\n"
                        "質問: {question}\n\n"
                        "回答のポイント:\n"
                        "- どのような情報収集プロセスを踏んでいるか\n"
                        "- 検索エンジンを使う場合、どんなキーワードを想定しているか\n"
                        "- 何を重視しているか（価格、口コミ、評判、機能、サポート等）\n"
                        "- 具体的な利用シーンや期待している効果\n"
                    ),
                ),
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

class SEOStrategyDocumentGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, interviews: list[Interview]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "あなたはSEO戦略を立案するプロフェッショナルです。"
                        "以下の情報に基づいて、具体的で効果的なSEO戦略を提案してください。"
                    ),
                ),
                (
                    "human",
                    (
                        "以下のWebサイト概要と複数の読者ペルソナ（インタビュー結果）に基づいて、"
                        "実践的なSEO戦略ドキュメントを作成してください。\n\n"
                        "【Webサイトの概要】\n{user_request}\n\n"
                        "【インタビュー結果】\n{interview_results}\n\n"
                        "最低限、以下の項目を盛り込んでください:\n"
                        "1. SEO施策の目的\n"
                        "2. ターゲット層と主要キーワード\n"
                        "3. 現状の課題\n"
                        "4. 施策の優先度とロードマップ（短期・中期・長期）\n"
                        "5. キーワード戦略やコンテンツ最適化の方針\n"
                        "6. リンクビルディングの戦略\n"
                        "7. 競合サイトの分析ポイント\n"
                        "8. 必要なツールやリソース、運用コストの目安\n"
                        "9. モニタリングと改善サイクル\n\n"
                        "プロが実践できるレベルの詳細な戦略ドキュメントを作成してください。"
                    ),
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
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
# The SEOAgent class with a state machine
# -----------------------------------------------------------------------------
class SEOAgent:
    def __init__(self, llm: ChatOpenAI, k: Optional[int] = None):
        self.persona_generator = PersonaGenerator(llm=llm, k=k)
        self.interview_conductor = InterviewConductor(llm=llm)
        self.seo_strategy_generator = SEOStrategyDocumentGenerator(llm=llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(InterviewState)
        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("generate_strategy", self._generate_strategy)

        # Entry point
        workflow.set_entry_point("generate_personas")

        # Edges
        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "generate_strategy")
        workflow.add_edge("generate_strategy", END)

        return workflow.compile()

    def _generate_personas(self, state: InterviewState) -> dict[str, Any]:
        new_personas = self.persona_generator.run(state.user_request)
        return {
            "personas": new_personas.personas,
            "iteration": state.iteration + 1,
        }

    def _conduct_interviews(self, state: InterviewState) -> dict[str, Any]:
        # If many personas, limit to last 5
        new_personas = state.personas[-5:]
        new_interviews = self.interview_conductor.run(state.user_request, new_personas)
        return {"interviews": new_interviews.interviews}

    def _generate_strategy(self, state: InterviewState) -> dict[str, Any]:
        seo_strategy_doc = self.seo_strategy_generator.run(
            state.user_request, state.interviews
        )
        return {"seo_strategy_doc": seo_strategy_doc}

    def run(self, user_request: str) -> str:
        initial_state = InterviewState(user_request=user_request)
        final_state = self.graph.invoke(initial_state)
        return final_state["seo_strategy_doc"]

# -----------------------------------------------------------------------------
# Cloud Run Functions HTTP entry point
# -----------------------------------------------------------------------------
@functions_framework.http
def main(request):
    # 1. Check referer
    if not check_referer(request):
        return (jsonify({'error': 'Unauthorized access'}), 403, create_cors_headers(request))

    # 2. CORS preflight check
    if request.method == 'OPTIONS':
        return ('', 204, create_cors_headers(request))

    try:
        # 3. Get JSON/args from request
        request_json = request.get_json(silent=True)
        request_args = request.args

        if request_json and 'user_request' in request_json:
            user_request = request_json['user_request']
            k = int(request_json.get('k', 3))
        elif request_args and 'user_request' in request_args:
            user_request = request_args['user_request']
            k = int(request_args.get('k', 3))
        else:
            return (
                jsonify({'error': 'user_request が提供されていません'}),
                400,
                create_cors_headers(request)
            )

        # 4. Initialize ChatOpenAI model
        #    (Change model_name to something valid for your environment)
        llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0.0)

        # 5. Run the SEO agent
        agent = SEOAgent(llm=llm, k=k)
        final_output = agent.run(user_request=user_request)

        response_data = {
            'result': final_output,
            'status': 'success'
        }
        return (jsonify(response_data), 200, create_cors_headers(request))

    except ValueError as ve:
        error_response = {
            'error': '無効な入力データ',
            'details': str(ve),
            'status': 'error'
        }
        return (jsonify(error_response), 400, create_cors_headers(request))

    except Exception as e:
        error_response = {
            'error': '内部サーバーエラー',
            'details': str(e),
            'traceback': traceback.format_exc(),
            'status': 'error'
        }
        return (jsonify(error_response), 500, create_cors_headers(request))
