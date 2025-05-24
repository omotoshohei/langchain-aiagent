import functions_framework
import traceback
from flask import jsonify
import operator
from typing import Annotated, Any, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

import os


# Data Model Definitions

class Persona(BaseModel):
    name: str = Field(..., description="Name of the persona")
    background: str = Field(..., description="Background of the persona")

class Personas(BaseModel):
    personas: list[Persona] = Field(
        default_factory=list, description="List of personas"
    )

class Interview(BaseModel):
    persona: Persona = Field(..., description="Target persona for the interview")
    question: str = Field(..., description="Question asked during the interview")
    answer: str = Field(..., description="Answer provided during the interview")

class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list, description="List of interview results"
    )

class EvaluationResult(BaseModel):
    reason: str = Field(..., description="Reason for the evaluation")
    is_sufficient: bool = Field(..., description="Whether the information is sufficient")

class InterviewState(BaseModel):
    user_request: str = Field(..., description="User's request")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list, description="List of generated personas"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list, description="List of conducted interviews"
    )
    requirements_doc: str = Field(default="", description="Generated requirements document")
    iteration: int = Field(
        default=0, description="Number of iterations for persona generation and interviews"
    )
    is_information_sufficient: bool = Field(
        default=False, description="Whether the information is sufficient"
    )

# Class Definitions for Various Generators

class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 5):
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert in creating diverse personas for user interviews.",
                ),
                (
                    "human",
                    f"Based on the following user request, generate {self.k} diverse personas for interviews.\n\n"
                    "User Request: {user_request}\n\n"
                    "Each persona should include a name and a brief background. Ensure diversity in age, gender, occupation, and technical expertise.",
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({"user_request": user_request})

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
                    "You are an expert in generating appropriate questions based on user requirements.",
                ),
                (
                    "human",
                    "Based on the following persona and user request, generate one question:\n\n"
                    "User Request: {user_request}\n"
                    "Persona: {persona_name} - {persona_background}\n\n"
                    "The question should be specific and designed to extract important information from the persona's perspective.",
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
                    "You are responding as the following persona: {persona_name} - {persona_background}",
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
            Interview(persona=persona, question=question, answer=answer)
            for persona, question, answer in zip(personas, questions, answers)
        ]

class InformationEvaluator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(EvaluationResult)

    def run(self, user_request: str, interviews: list[Interview]) -> EvaluationResult:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert in evaluating the sufficiency of information for comprehensive requirements documents.",
                ),
                (
                    "human",
                    "Based on the following user request and interview results, determine if sufficient information has been gathered to create a comprehensive requirements document.\n\n"
                    "User Request: {user_request}\n\n"
                    "Interview Results:\n{interview_results}",
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"Persona: {i.persona.name} - {i.persona.background}\n"
                    f"Question: {i.question}\nAnswer: {i.answer}\n"
                    for i in interviews
                ),
            }
        )

class RequirementsDocumentGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, interviews: list[Interview]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert in creating requirements documents based on collected information.",
                ),
                (
                    "human",
                    "Based on the following user request and multiple personas' interview results, create a requirements document.\n\n"
                    "User Request: {user_request}\n\n"
                    "Interview Results:\n{interview_results}\n"
                    "The requirements document should include the following sections:\n"
                    "1. Project Overview\n"
                    "2. Key Features\n"
                    "3. Non-functional Requirements\n"
                    "4. Constraints\n"
                    "5. Target Users\n"
                    "6. Priorities\n"
                    "7. Risks and Mitigation Strategies\n",
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"Persona: {i.persona.name} - {i.persona.background}\n"
                    f"Question: {i.question}\nAnswer: {i.answer}\n"
                    for i in interviews
                ),
            }
        )

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

@functions_framework.http
def main(request):
    if not check_referer(request):
        return (jsonify({'error': 'Unauthorized access'}), 403, create_cors_headers(request))
    # CORS preflight check
    if request.method == 'OPTIONS':
        headers = create_cors_headers(request)
        return ('', 204, headers)

    try:
        # Extract request data
        request_json = request.get_json(silent=True)
        request_args = request.args

        # Retrieve user_request
        if request_json and 'user_request' in request_json:
            user_request = request_json['user_request']
            k = int(request_json.get('k', 3))
        elif request_args and 'user_request' in request_args:
            user_request = request_args['user_request']
            k = int(request_args.get('k', 3))
        else:
            return (jsonify({'error': 'user_request is not provided'}), 400, create_cors_headers(request))

        # Initialize ChatOpenAI model
        llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0.0)

        # Initialize the Documentation Agent
        agent = DocumentationAgent(llm=llm, k=k)

        # Run the agent and get the final output
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
