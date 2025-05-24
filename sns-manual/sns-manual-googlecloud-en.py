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
    name: str = Field(..., description="Name of the persona")
    background: str = Field(..., description="Background of the persona")

# ペルソナのリストを表すデータモデル
class Personas(BaseModel):
    personas: list[Persona] = Field(
        default_factory=list, description="List of personas"
    )

# インタビュー内容を表すデータモデル
class Interview(BaseModel):
    persona: Persona = Field(..., description="Persona targeted for the interview")
    question: str = Field(..., description="Question in the interview")
    answer: str = Field(..., description="Answer in the interview")

# インタビュー結果のリストを表すデータモデル
class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list, description="List of interview results"
    )

# 要件定義生成AIエージェントのステート
class InterviewState(BaseModel):
    user_request: str = Field(..., description="Request from the user")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list, description="List of generated personas"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list, description="List of conducted interviews"
    )
    requirements_doc: str = Field(default="", description="Generated requirement definition")
    iteration: int = Field(
        default=0, description="Number of iterations for persona generation and interviews"
    )
    # 情報評価ステップ削除に伴い、is_information_sufficient フィールド等も削除

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
        # プロンプトテンプレートを定義
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert in creating diverse personas for interviews targeting SNS account users.",
                ),
                (
                    "human",
                    f"Generate {self.k} diverse personas for interviews on the following SNS topic:\n\n"
                    "Topic: {user_request}\n\n"
                    "Include the name and brief background for each reader persona. Ensure diversity in age, gender, occupation, and knowledge level about the topic.",
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
        # 3. インタビューのリストを作成
        interviews = self._create_interviews(personas, questions, answers)
        return InterviewResult(interviews=interviews)

    def _generate_questions(
        self, user_request: str, personas: list[Persona]
    ) -> list[str]:
        question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an interviewer. Create questions to draw out the concerns and challenges of the persona.",
                ),
                (
                    "human",
                    "Create one open-ended question that allows the following reader persona to discuss their concerns and challenges regarding the SNS topic:\n\n"
                    "Topic: {user_request}\n"
                    "Reader Persona: {persona_name} - {persona_background}\n\n"
                    "Keep the question simple so that this persona can openly discuss their concerns.",
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
                    "You are the following reader persona. Please specifically describe your concerns and challenges in response to the interviewer's question.\n\nPersona: {persona_name} - {persona_background}",
                ),
                ("human", "Question: {question}"),
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

# 要件定義書を生成するクラス
class RequirementsDocumentGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, interviews: list[Interview]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert in creating SNS operation manuals based on collected information.",
                ),
                (
                    "human",
                    "Create an SNS operation manual based on the following topic and interview results from multiple reader personas.\n\n"
                    "Topic: {user_request}\n\n"
                    "Interview Results:\n{interview_results}\n"
                    "Include the following sections in the manual:\n"
                    "1. Purpose of SNS operation\n"
                    "2. Target audience\n"
                    "3. Audience concerns\n"
                    "4. Topics and hashtags\n"
                    "5. SNS platforms, posting frequency, timing, and formats\n"
                    "6. Post themes\n"
                    "7. Important notes\n\n"
                    "Ensure the output is in English.\n\nInstruction for writing the article:",
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

        # 遷移設定（評価ステップ無し）
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
        # ペルソナが多い場合は最後の5人のみに絞る
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
ALLOWED_ORIGINS = [
    "https://arigatoai.com",
    "https://www.arigatoai.com",
    "https://heysho.com",
    "https://www.heysho.com",
    # 開発環境で http://localhost:3000 等を許可したい場合は追加
    # "http://localhost:3000",
]
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

# -----------------------------------------------------------------------------
# Cloud Run Functionsエントリーポイント
# -----------------------------------------------------------------------------
@functions_framework.http
def main(request):
    # リファラーのチェック（必要に応じて）
    if not check_referer(request):
        return (jsonify({'error': 'Unauthorized access'}), 403, create_cors_headers(request))

    # CORSのプリフライトチェック
    if request.method == 'OPTIONS':
        headers = create_cors_headers(request)
        return ('', 204, headers)

    try:
        # リクエストデータの取得
        request_json = request.get_json(silent=True)
        request_args = request.args

        # user_request と k を取得
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
                create_cors_headers(request)
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
