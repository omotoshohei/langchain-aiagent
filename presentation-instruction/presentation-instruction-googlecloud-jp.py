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

########################################
# データモデル
########################################

# ペルソナを表すデータモデル
class Persona(BaseModel):
    name: str = Field(..., description="ペルソナの名前")
    background: str = Field(..., description="ペルソナ（ターゲット）の持つ背景や特徴")

# ペルソナのリストを表すデータモデル
class Personas(BaseModel):
    personas: list[Persona] = Field(
        default_factory=list, description="ペルソナのリスト（複数のターゲット）"
    )

# インタビュー内容を表すデータモデル
class Interview(BaseModel):
    persona: Persona = Field(..., description="インタビュー対象のペルソナ")
    question: str = Field(..., description="インタビュアーからの質問")
    answer: str = Field(..., description="ペルソナの回答")

# インタビュー結果のリストを表すデータモデル
class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list, description="インタビュー結果のリスト"
    )

# プレゼン要件定義 生成AIエージェント用 State
# （InformationEvaluatorを省略するため、iteration/is_information_sufficientを削除）
class PresentationInterviewState(BaseModel):
    user_request: str = Field(..., description="ユーザーが作りたいプレゼンの概要/要求")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list, description="生成されたペルソナのリスト"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list, description="実施されたインタビューのリスト"
    )
    requirements_doc: str = Field(
        default="", description="最終生成されるプレゼン要件定義書"
    )

########################################
# PersonaGenerator
########################################
class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 5):
        # 出力を Personals モデルにパース
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "あなたはプレゼンテーションのターゲット分析を行う専門家です。"),
                (
                    "human",
                    f"以下のプレゼンのトピックに関するターゲットリサーチをするため、{self.k}人分のペルソナを生成してください。\n\n"
                    "【プレゼンのトピック】\n{user_request}\n\n"
                    "各ペルソナには、名前と簡単な背景（年齢、役職、プレゼンテーマとの関わり、知識レベルなど）を含め、\n"
                    "ターゲットの多様性が高まるように工夫してください。"
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({"user_request": user_request})

########################################
# InterviewConductor
########################################
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
                ("system", "あなたはインタビュアーです。ペルソナが抱える悩みや知りたい情報を引き出すための質問を作ります。"),
                (
                    "human",
                    "以下のプレゼン内容を踏まえて、ペルソナが思う課題や期待をうまく引き出せるオープンな質問を1つ作成してください。\n\n"
                    "【プレゼンのトピック】\n{user_request}\n"
                    "【ペルソナ】\n名前：{persona_name}\n背景：{persona_background}\n\n"
                    "質問はシンプルかつ会話を拡張できる形にしてください。"
                ),
            ]
        )
        question_chain = question_prompt | self.llm | StrOutputParser()

        question_queries = [
            {
                "user_request": user_request,
                "persona_name": p.name,
                "persona_background": p.background,
            }
            for p in personas
        ]
        return question_chain.batch(question_queries)

    def _generate_answers(self, personas: list[Persona], questions: list[str]) -> list[str]:
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは以下の読者ペルソナ（プレゼンのターゲット）です。質問に対してあなたの本音を答えてください。\n\n"
                    "ペルソナ情報:\n名前：{persona_name}\n背景：{persona_background}"
                ),
                ("human", "質問：{question}")
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

    def _create_interviews(self, personas: list[Persona], questions: list[str], answers: list[str]) -> list[Interview]:
        return [
            Interview(persona=persona, question=q, answer=a)
            for persona, q, a in zip(personas, questions, answers)
        ]

########################################
# RequirementsDocumentGenerator
########################################
class RequirementsDocumentGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, interviews: list[Interview]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "あなたはプレゼンテーションの要件定義書を作成するアシスタントです。"),
                (
                    "human",
                    "以下の【プレゼンのトピック】と【インタビュー結果】を参考に、\n"
                    "「プレゼン要件定義書」を作成してください。\n"
                    "必ず以下のようなセクションに分けて日本語で作成してください:\n\n"
                    "1. プレゼンの目的（何を達成したいか）\n"
                    "2. ターゲットの特徴（誰向けのプレゼンか。知識レベルや立場、目的、期待など）\n"
                    "3. ターゲットの課題やニーズ\n"
                    "4. プレゼンで扱うトピックの要点（重要ポイント、データ、説得力を持たせる材料など）\n"
                    "5. プレゼンの形式・構成案（スライド枚数、時間配分、資料、デモの有無など）\n"
                    "6. ターゲットに与えたいインパクト（どういう反応やアクションを期待するか）\n"
                    "7. 想定されるQ&Aや懸念点への対策\n\n"
                    "【プレゼンのトピック】\n{user_request}\n\n"
                    "【インタビュー結果】\n{interview_results}\n\n"
                    "上記を踏まえ、プレゼン要件定義書を作成してください。"
                ),
            ]
        )

        interview_text = "\n".join(
            f"▼ペルソナ: {i.persona.name} - {i.persona.background}\n"
            f"   質問: {i.question}\n"
            f"   回答: {i.answer}\n"
            for i in interviews
        )

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "user_request": user_request,
            "interview_results": interview_text
        })

########################################
# プレゼン要件定義エージェント （InformationEvaluatorを省略）
########################################
class PresentationDocumentationAgent:
    def __init__(self, llm: ChatOpenAI, k: Optional[int] = 3):
        self.persona_generator = PersonaGenerator(llm=llm, k=k)
        self.interview_conductor = InterviewConductor(llm=llm)
        self.requirements_generator = RequirementsDocumentGenerator(llm=llm)

        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """
        StateGraphの定義（情報評価ステップは省略）。
        1) ペルソナ生成 → 2) インタビュー → 3) 要件定義書生成 → 終了
        """
        workflow = StateGraph(PresentationInterviewState)

        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("generate_requirements", self._generate_requirements)

        workflow.set_entry_point("generate_personas")

        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "generate_requirements")
        workflow.add_edge("generate_requirements", END)

        return workflow.compile()

    def _generate_personas(self, state: PresentationInterviewState) -> dict[str, Any]:
        new_personas: Personas = self.persona_generator.run(state.user_request)
        return {
            "personas": state.personas + new_personas.personas
        }

    def _conduct_interviews(self, state: PresentationInterviewState) -> dict[str, Any]:
        # ペルソナが多い場合は最後の5人のみに絞る
        recent_personas = state.personas[-5:]
        interview_result: InterviewResult = self.interview_conductor.run(
            state.user_request, recent_personas
        )
        return {
            "interviews": state.interviews + interview_result.interviews
        }

    def _generate_requirements(self, state: PresentationInterviewState) -> dict[str, Any]:
        requirements_doc: str = self.requirements_generator.run(
            state.user_request, state.interviews
        )
        return {"requirements_doc": requirements_doc}

    def run(self, user_request: str) -> str:
        initial_state = PresentationInterviewState(user_request=user_request)
        final_state = self.graph.invoke(initial_state)
        return final_state["requirements_doc"]

########################################
# CORS関連ヘルパー
########################################
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

########################################
# Cloud Run Functionsエントリーポイント
########################################
@functions_framework.http
def main(request):
    # リファラーのチェック
    if not check_referer(request):
        return (jsonify({'error': 'Unauthorized access'}), 403, create_cors_headers())

    # CORSプリフライトリクエストに対応
    if request.method == 'OPTIONS':
        return ('', 204, create_cors_headers())

    try:
        request_json = request.get_json(silent=True)
        request_args = request.args

        if request_json:
            # JSONボディから5つの入力項目を取得
            presentation_topic = request_json.get('presentation_topic', '')
            time_minutes = request_json.get('time_minutes', '')
            slide_count = request_json.get('slide_count', '')
            target_type = request_json.get('target_type', '')
            knowledge_level = request_json.get('knowledge_level', '')
            k = int(request_json.get('k', 3))  # ペルソナ人数 (デフォルト3)

        elif request_args:
            # クエリパラメータから5つの入力項目を取得
            presentation_topic = request_args.get('presentation_topic', '')
            time_minutes = request_args.get('time_minutes', '')
            slide_count = request_args.get('slide_count', '')
            target_type = request_args.get('target_type', '')
            knowledge_level = request_args.get('knowledge_level', '')
            k = int(request_args.get('k', 3))

        else:
            return (jsonify({'error': 'パラメータが提供されていません'}), 400, create_cors_headers())

        # ユーザー入力を1つの文字列にまとめる
        user_request = (
            f"【プレゼンの目的/トピック】: {presentation_topic}\n"
            f"【与えられた時間(分)】: {time_minutes}\n"
            f"【スライド枚数】: {slide_count}\n"
            f"【ターゲット】: {target_type}\n"
            f"【ターゲットの知識レベル】: {knowledge_level}"
        )

        # ChatOpenAIモデル初期化 (モデル名などは必要に応じて変更)
        llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0.0)

        # エージェントを初期化して実行
        agent = PresentationDocumentationAgent(llm=llm, k=k)
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
