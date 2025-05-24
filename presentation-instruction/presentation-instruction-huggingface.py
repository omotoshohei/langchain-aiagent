import os
import streamlit as st

# If you have secrets set in your Hugging Face Space, you can load them:
#   1. Go to your Space's "Settings" > "Repository secrets"
#   2. Add OPENAI_API_KEY, LANGCHAIN_API_KEY, etc.
#   3. Then call them in your code like st.secrets["OPENAI_API_KEY"]

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "LANGCHAIN_API_KEY" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

# Optional: If you need these environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "agent-presentation"

from typing import Annotated, Any, Optional
import operator
from pydantic import BaseModel, Field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# -----------------------------
# Data Models
# -----------------------------
class Persona(BaseModel):
    name: str = Field(..., description="Persona's name")
    background: str = Field(..., description="Persona's (target's) background or characteristics")

class Personas(BaseModel):
    personas: list[Persona] = Field(
        default_factory=list, description="A list of personas (multiple targets)"
    )

class Interview(BaseModel):
    persona: Persona = Field(..., description="The persona being interviewed")
    question: str = Field(..., description="The interviewer's question")
    answer: str = Field(..., description="The persona's answer")

class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list, description="A list of interview results"
    )

class PresentationInterviewState(BaseModel):
    user_request: str = Field(..., description="A summary/requirement of the presentation the user wants to create")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list, description="A list of generated personas"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list, description="The list of interviews conducted"
    )
    requirements_doc: str = Field(
        default="", description="The final generated presentation requirements document"
    )
    iteration: int = Field(default=0, description="Number of iterations repeating persona generation and interviews")

# -----------------------------
# Persona Generator
# -----------------------------
class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 5):
        # Using structured output ensures we parse a Personas object
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert in analyzing target audiences for presentations."
                ),
                (
                    "human",
                    f"Please generate {self.k} personas in order to conduct target research on the following presentation topic.\n\n"
                    "[Presentation Topic]\n{{user_request}}\n\n"
                    "For each persona, include their name and a brief background (age, role, relationship to the presentation topic, knowledge level, etc.), "
                    "and make sure to maximize the diversity of the target audience."
                ),
            ]
        )
        chain = prompt | self.llm
        return chain.invoke({"user_request": user_request})

# -----------------------------
# Interview Conductor
# -----------------------------
class InterviewConductor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, personas: list[Persona]) -> InterviewResult:
        questions = self._generate_questions(user_request=user_request, personas=personas)
        answers = self._generate_answers(personas=personas, questions=questions)
        interviews = self._create_interviews(personas, questions, answers)
        return InterviewResult(interviews=interviews)

    def _generate_questions(self, user_request: str, personas: list[Persona]) -> list[str]:
        question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an interviewer. Create questions that draw out the challenges or concerns of each persona."
                ),
                (
                    "human",
                    "Based on the following presentation topic, please create **one open-ended question** that draws out "
                    "the persona's issues or expectations.\n\n"
                    "[Presentation Topic]\n{user_request}\n"
                    "[Persona]\nName: {persona_name}\nBackground: {persona_background}\n\n"
                    "Please keep the question simple and let it expand the conversation."
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
                    "You are the following reader persona (the target of the presentation). "
                    "Please answer the question from your honest perspective.\n\n"
                    "Persona info:\nName: {persona_name}\nBackground: {persona_background}"
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
        interviews = []
        for persona, q, a in zip(personas, questions, answers):
            interviews.append(Interview(persona=persona, question=q, answer=a))
        return interviews

# -----------------------------
# Requirements Document Generator
# -----------------------------
class RequirementsDocumentGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, interviews: list[Interview]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an assistant that creates presentation requirement documents."
                ),
                (
                    "human",
                    "Based on the following [Presentation Topic] and [Interview Results], please create a 'Presentation Requirements Document'.\n"
                    "Make sure to write it in English and include the following sections:\n\n"
                    "1. Purpose of the Presentation (What do we want to achieve?)\n"
                    "2. Characteristics of the Target Audience (Who is it for? Knowledge level, roles, motivations, expectations, etc.)\n"
                    "3. Challenges or Needs of the Target Audience\n"
                    "4. Key Points of the Presentation Topic (important points, data, persuasive material, etc.)\n"
                    "5. Presentation Format and Structure (number of slides, time allocation, materials, demos, etc.)\n"
                    "6. Desired Impact on the Audience (What reaction or action do we expect?)\n"
                    "7. Anticipated Q&A or Concerns, and Countermeasures\n\n"
                    "[Presentation Topic]\n{user_request}\n\n"
                    "[Interview Results]\n{interview_results}\n\n"
                    "Please incorporate these details into the presentation requirements document."
                ),
            ]
        )

        interview_text = ""
        for i in interviews:
            interview_text += (
                f"▼ Persona: {i.persona.name} - {i.persona.background}\n"
                f"   Question: {i.question}\n"
                f"   Answer: {i.answer}\n\n"
            )

        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({
            "user_request": user_request,
            "interview_results": interview_text
        })

# -----------------------------
# Presentation Documentation Agent
# -----------------------------
class PresentationDocumentationAgent:
    def __init__(self, llm: ChatOpenAI, k: Optional[int] = 3):
        self.persona_generator = PersonaGenerator(llm=llm, k=k)
        self.interview_conductor = InterviewConductor(llm=llm)
        self.requirements_generator = RequirementsDocumentGenerator(llm=llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """
        The StateGraph is defined in the following order:
        1) Generate personas → 2) Conduct interviews → 3) Generate the requirements document → Done
        (Skipping the step to evaluate information sufficiency)
        """
        workflow = StateGraph(PresentationInterviewState)

        # Nodes
        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("generate_requirements", self._generate_requirements)

        # Entry point
        workflow.set_entry_point("generate_personas")

        # Transitions
        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "generate_requirements")
        workflow.add_edge("generate_requirements", END)

        return workflow.compile()

    def _generate_personas(self, state: PresentationInterviewState) -> dict[str, Any]:
        new_personas: Personas = self.persona_generator.run(state.user_request)
        return {
            "personas": state.personas + new_personas.personas,
            "iteration": state.iteration + 1
        }

    def _conduct_interviews(self, state: PresentationInterviewState) -> dict[str, Any]:
        # Conduct interviews on the most recently added personas (last 5)
        new_personas = state.personas[-5:]
        interview_result: InterviewResult = self.interview_conductor.run(
            state.user_request, new_personas
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

# -----------------------------
# Streamlit App
# -----------------------------
def main():
    st.title("Presentation Requirements Agent")

    # Initialize model (adjust model_name if needed)
    # llm = ChatOpenAI(model_name="gpt-4o-2024-11-20", temperature=0.0)
    llm = ChatOpenAI(model_name="gpt-4.1-mini-2025-04-14", temperature=0.3)
    # llm = ChatOpenAI(model_name="gpt-4.1-2025-04-14", temperature=0.3)
    agent = PresentationDocumentationAgent(llm=llm, k=3)

    # User inputs
    presentation_topic = st.text_input("1. Enter the purpose or topic of the presentation you want to create:")
    time_minutes = st.text_input("2. Enter the allotted time (in minutes):")
    slide_count = st.text_input("3. Enter the number of slides:")
    target_type = st.selectbox(
        "4. Select the target audience:", ["Boss", "Client", "Colleague"]
    )
    knowledge_level = st.selectbox(
        "5. Select the audience's knowledge level about the topic:",
        ["Beginner", "Some experience", "Deep expertise"],
    )

    if st.button("Generate Presentation Requirements Document"):
        # Build the user_request text
        with st.spinner("Generating ... Please wait for 40 seconds"):

            user_request = (
                f"[Presentation Purpose/Topic]: {presentation_topic}\n"
                f"[Allotted Time (minutes)]: {time_minutes}\n"
                f"[Number of Slides]: {slide_count}\n"
                f"[Target Audience]: {target_type}\n"
                f"[Knowledge Level]: {knowledge_level}"
            )

            # Run the agent
            final_output = agent.run(user_request=user_request)

        st.subheader("----- Generated Presentation Requirements Document -----")
        st.write(final_output)

# This is the entry point for Streamlit
if __name__ == "__main__":
    main()
