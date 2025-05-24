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
# Data Models
########################################

# Data model representing a persona
class Persona(BaseModel):
    name: str = Field(..., description="Name of the persona")
    background: str = Field(..., description="Background and characteristics of the persona (target)")

# Data model representing a list of personas
class Personas(BaseModel):
    personas: list[Persona] = Field(
        default_factory=list, description="List of personas (multiple targets)"
    )

# Data model representing interview content
class Interview(BaseModel):
    persona: Persona = Field(..., description="Persona being interviewed")
    question: str = Field(..., description="Question from the interviewer")
    answer: str = Field(..., description="Answer from the persona")

# Data model representing a list of interview results
class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list, description="List of interview results"
    )

# State for Presentation Requirements Definition Generation AI Agent
# (Omitting InformationEvaluator, iteration/is_information_sufficient removed)
class PresentationInterviewState(BaseModel):
    user_request: str = Field(..., description="Overview/requirements of the presentation the user wants to create")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list, description="List of generated personas"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list, description="List of conducted interviews"
    )
    requirements_doc: str = Field(
        default="", description="The final generated presentation requirements document"
    )

########################################
# PersonaGenerator
########################################
class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 5):
        # Parse output into Personas model
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert in analyzing presentation targets."),
                (
                    "human",
                    f"To conduct target research for the following presentation topic, generate {self.k} personas.\n\n"
                    "【Presentation Topic】\n{user_request}\n\n"
                    "For each persona, include their name and a brief background (age, position, relation to the presentation theme, knowledge level, etc.).\n"
                    "Ensure diversity among the targets."
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
                ("system", "You are an interviewer. Create questions to uncover the problems and information needs of the persona."),
                (
                    "human",
                    "Based on the following presentation topic, create an open-ended question that can effectively draw out the persona's concerns and expectations.\n\n"
                    "【Presentation Topic】\n{user_request}\n"
                    "【Persona】\nName: {persona_name}\nBackground: {persona_background}\n\n"
                    "Make the question simple and capable of expanding the conversation."
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
                    "You are the following target persona for the presentation. Answer the question honestly with your true thoughts."
                    "\n\nPersona Information:\nName: {persona_name}\nBackground: {persona_background}"
                ),
                ("human", "Question: {question}")
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
                ("system", "You are an assistant that creates presentation requirements documents."),
                (
                    "human",
                    "Based on the following 【Presentation Topic】 and 【Interview Results】, "
                    "create a 'Presentation Requirements Document'.\n"
                    "Be sure to divide the document into sections as follows and write in English:\n\n"
                    "1. Purpose of the presentation (what you want to achieve)\n"
                    "2. Characteristics of the target (who the presentation is for; knowledge level, position, purpose, expectations, etc.)\n"
                    "3. Challenges and needs of the target\n"
                    "4. Key points of the presentation topic (important points, data, persuasive materials, etc.)\n"
                    "5. Proposed format/structure of the presentation (number of slides, time allocation, materials, demos, etc.)\n"
                    "6. Impact desired on the target (what reaction or action is expected)\n"
                    "7. Anticipated Q&A and countermeasures for concerns\n\n"
                    "【Presentation Topic】\n{user_request}\n\n"
                    "【Interview Results】\n{interview_results}\n\n"
                    "Based on the above, create the presentation requirements documen."
                ),
            ]
        )

        interview_text = "\n".join(
            f"▼Persona: {i.persona.name} - {i.persona.background}\n"
            f"   Question: {i.question}\n"
            f"   Answer: {i.answer}\n"
            for i in interviews
        )

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "user_request": user_request,
            "interview_results": interview_text
        })

########################################
# Presentation Requirements Definition Agent (Omitting InformationEvaluator)
########################################
class PresentationDocumentationAgent:
    def __init__(self, llm: ChatOpenAI, k: Optional[int] = 3):
        self.persona_generator = PersonaGenerator(llm=llm, k=k)
        self.interview_conductor = InterviewConductor(llm=llm)
        self.requirements_generator = RequirementsDocumentGenerator(llm=llm)

        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """
        Definition of the StateGraph (omitting the information evaluation step).
        1) Generate personas → 2) Conduct interviews → 3) Generate requirements document → End
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
        # If there are many personas, limit to the last 5
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
# CORS Helpers
########################################
def create_cors_headers():
    return {
        'Access-Control-Allow-Origin': 'https://arigatoai.com',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '600',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }

def check_referer(request):
    referer = request.headers.get('Referer', '')
    allowed_domains = ['arigatoai.com', 'www.arigatoai.com']
    return any(domain in referer for domain in allowed_domains)

########################################
# Cloud Run Functions Entry Point
########################################
@functions_framework.http
def main(request):
    # Referrer check
    if not check_referer(request):
        return (jsonify({'error': 'Unauthorized access'}), 403, create_cors_headers())

    # CORS preflight request handling
    if request.method == 'OPTIONS':
        return ('', 204, create_cors_headers())

    try:
        request_json = request.get_json(silent=True)
        request_args = request.args

        if request_json:
            # Retrieve five input items from JSON body
            presentation_topic = request_json.get('presentation_topic', '')
            time_minutes = request_json.get('time_minutes', '')
            slide_count = request_json.get('slide_count', '')
            target_type = request_json.get('target_type', '')
            knowledge_level = request_json.get('knowledge_level', '')
            k = int(request_json.get('k', 3))  # Number of personas (default 3)

        elif request_args:
            # Retrieve five input items from query parameters
            presentation_topic = request_args.get('presentation_topic', '')
            time_minutes = request_args.get('time_minutes', '')
            slide_count = request_args.get('slide_count', '')
            target_type = request_args.get('target_type', '')
            knowledge_level = request_args.get('knowledge_level', '')
            k = int(request_args.get('k', 3))

        else:
            return (jsonify({'error': 'No parameters provided'}), 400, create_cors_headers())

        # Combine user inputs into one string
        user_request = (
            f"【Purpose/Topic of Presentation】: {presentation_topic}\n"
            f"【Allocated Time (minutes)】: {time_minutes}\n"
            f"【Number of Slides】: {slide_count}\n"
            f"【Target Audience】: {target_type}\n"
            f"【Knowledge Level of Target】: {knowledge_level}"
        )

        # Initialize ChatOpenAI model (modify model name as needed)
        llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0.0)

        # Initialize and execute agent
        agent = PresentationDocumentationAgent(llm=llm, k=k)
        final_output = agent.run(user_request=user_request)

        response_data = {
            'result': final_output,
            'status': 'success'
        }
        return (jsonify(response_data), 200, create_cors_headers())

    except ValueError as ve:
        error_response = {
            'error': 'Invalid input data',
            'details': str(ve),
            'status': 'error'
        }
        return (jsonify(error_response), 400, create_cors_headers())

    except Exception as e:
        error_response = {
            'error': 'Internal server error',
            'details': str(e),
            'traceback': traceback.format_exc(),
            'status': 'error'
        }
        return (jsonify(error_response), 500, create_cors_headers())
