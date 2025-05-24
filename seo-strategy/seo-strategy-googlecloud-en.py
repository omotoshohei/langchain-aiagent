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
# CORS / Referer checking helpers
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
# Data Models
# -----------------------------------------------------------------------------
class Persona(BaseModel):
    name: str = Field(..., description="Persona name")
    background: str = Field(..., description="Persona background information")

class Personas(BaseModel):
    personas: list[Persona] = Field(
        default_factory=list, description="List of personas"
    )

class Interview(BaseModel):
    persona: Persona = Field(..., description="Interview target persona")
    question: str = Field(..., description="Question asked during the interview")
    answer: str = Field(..., description="Answer provided during the interview")

class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list, description="List of interview results"
    )

class InterviewState(BaseModel):
    user_request: str = Field(..., description="User request")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list, description="List of generated personas"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list, description="List of conducted interviews"
    )
    seo_strategy_doc: str = Field(default="", description="Generated SEO strategy document")
    iteration: int = Field(
        default=0, description="Iteration count of persona generation and interviews"
    )

# -----------------------------------------------------------------------------
# Classes for each step
# -----------------------------------------------------------------------------
class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 5):
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        # --- Prompt tailored for SEO ---
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert specialized in creating personas of potential customers who may be interested in the service provided by the client. "
                    "Clearly indicate the specific problems each persona faces and what information they are seeking."
                ),
                (
                    "human",
                    (
                        f"Based on the following service overview, generate {self.k} diverse personas.\n\n"
                        "【Service Overview】\n{user_request}\n\n"
                        "Each persona should include:\n"
                        "- Name\n"
                        "- Age, gender, occupation\n"
                        "- What challenges or problems they face\n"
                        "- Likely sources where they would find your service (search engines, social media, referrals, etc.)\n"
                        "- What information they particularly focus on (price, reviews, feature comparisons, etc.)\n"
                        "- What keywords they might use to search (estimate is fine)\n"
                        "Please create realistic and concrete profiles."
                    ),
                ),
            ]
        )
        # --- End of SEO-tailored prompt ---
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
                    "You are an interviewer. Your job is to delve into potential customers' problems and challenges and discover "
                    "how they find, select, and intend to use the service."
                ),
                (
                    "human",
                    (
                        "Please create an open-ended question so that the following persona can honestly discuss their problems and "
                        "information gathering (including search engines and social media) regarding your service.\n\n"
                        "【Service Overview】\n{user_request}\n\n"
                        "【Persona】\n{persona_name} - {persona_background}\n\n"
                        "Points for the question:\n"
                        "- Elicit specific problems and goals\n"
                        "- Clarify where they gather information\n"
                        "- Understand what they emphasize when comparing options\n"
                        "Keep it simple yet thought-provoking."
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
                    "You are the following persona. Answer honestly about the specific problems you have regarding the service, "
                    "how you look for information, and what criteria you use to decide."
                ),
                (
                    "human",
                    (
                        "Persona: {persona_name} - {persona_background}\n\n"
                        "Question: {question}\n\n"
                        "Points for the answer:\n"
                        "- Describe your information gathering process\n"
                        "- If using search engines, what keywords might you use\n"
                        "- What factors do you focus on (price, reviews, reputation, features, support, etc.)\n"
                        "- Specific use cases and expected outcomes\n"
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
                    "You are a professional who devises SEO strategies. Based on the following information, propose a concrete and effective SEO strategy."
                ),
                (
                    "human",
                    (
                        "Using the website overview and multiple reader personas (interview results) below, "
                        "create a practical SEO strategy document in English.\n\n"
                        "【Website Overview】\n{user_request}\n\n"
                        "【Interview Results】\n{interview_results}\n\n"
                        "At minimum, include the following items:\n"
                        "1. Purpose of SEO measures (desired KPIs)\n"
                        "2. Target audience and key keywords\n"
                        "3. Current challenges (technical SEO, content, backlinks, etc.)\n"
                        "4. Prioritization and roadmap (short-, mid-, long-term)\n"
                        "5. Keyword strategy and content optimization policy\n"
                        "6. Link building strategy\n"
                        "7. Points for competitor analysis\n"
                        "8. Necessary tools, resources, estimated operational costs\n"
                        "9. Monitoring and improvement cycle\n\n"
                        "Create a detailed strategy document at a professional level."
                    ),
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        interview_results_text = "\n".join(
            f"Persona: {i.persona.name} - {i.persona.background}\n"
            f"Question: {i.question}\nAnswer: {i.answer}\n"
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
                jsonify({'error': 'user_request is not provided'}),
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
