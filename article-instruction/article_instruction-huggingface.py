import os
import streamlit as st

# ⬇️ If you have secrets set in your Hugging Face Space:
#   1. Go to → Settings → “Repository secrets” on your Space
#   2. Add OPENAI_API_KEY, LANGCHAIN_API_KEY, etc.
#   3. They will be accessible from st.secrets

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
if "LANGCHAIN_API_KEY" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

# LangSmith / LangChain env (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "article-instruction"

from typing import Annotated, Any, Optional
import operator
from pydantic import BaseModel, Field

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

# -----------------------------------------------------------------------------
# 1. Data Models
# -----------------------------------------------------------------------------
class Persona(BaseModel):
    name: str = Field(..., description="Persona name")
    background: str = Field(..., description="Brief background (age, role, etc.)")

class Personas(BaseModel):
    personas: list[Persona] = Field(default_factory=list, description="List of personas")

class Interview(BaseModel):
    persona: Persona = Field(...)
    question: str = Field(...)
    answer: str = Field(...)

class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(default_factory=list)

class ArticleInterviewState(BaseModel):
    user_request: str = Field(..., description="The article topic & meta info from the user")
    personas: Annotated[list[Persona], operator.add] = Field(default_factory=list)
    interviews: Annotated[list[Interview], operator.add] = Field(default_factory=list)
    requirements_doc: str = Field(default="")
    iteration: int = Field(default=0)

# -----------------------------------------------------------------------------
# 2. Persona Generator
# -----------------------------------------------------------------------------
class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 5):
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert at defining reader personas for blog articles."
            ),
            (
                "human",
                "Generate {k} diverse reader personas for the following article topic.\n\n"
                "[Article Topic]\n{{user_request}}\n\n"
                "For each persona include: name, brief background (age, profession), knowledge level, primary search intent / motivation, and preferred device (mobile/desktop)."
            ),
        ])
        chain = prompt | self.llm
        return chain.invoke({"user_request": user_request, "k": self.k})

# -----------------------------------------------------------------------------
# 3. Interview Conductor
# -----------------------------------------------------------------------------
class InterviewConductor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, personas: list[Persona]) -> InterviewResult:
        questions = self._generate_questions(user_request, personas)
        answers = self._generate_answers(personas, questions)
        interviews = [Interview(persona=p, question=q, answer=a) for p, q, a in zip(personas, questions, answers)]
        return InterviewResult(interviews=interviews)

    def _generate_questions(self, user_request: str, personas: list[Persona]) -> list[str]:
        q_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an interviewer tasked with uncovering the reader's real concerns and expectations related to the blog topic."
            ),
            (
                "human",
                "Given the blog topic below, write **one open‑ended question** that would reveal this persona's pain points or desired outcome.\n\n"
                "[Article Topic]\n{user_request}\n"
                "[Persona Info]\nName: {persona_name}\nBackground: {persona_background}"
            ),
        ])
        chain = q_prompt | self.llm | StrOutputParser()
        return chain.batch([
            {"user_request": user_request, "persona_name": p.name, "persona_background": p.background}
            for p in personas
        ])

    def _generate_answers(self, personas: list[Persona], questions: list[str]) -> list[str]:
        a_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Answer the question as the persona described below, sharing concrete struggles, hopes, and expectations."
            ),
            (
                "human",
                "Persona: {persona_name} / {persona_background}\nQuestion: {question}"
            ),
        ])
        chain = a_prompt | self.llm | StrOutputParser()
        return chain.batch([
            {"persona_name": p.name, "persona_background": p.background, "question": q}
            for p, q in zip(personas, questions)
        ])

# -----------------------------------------------------------------------------
# 4. Requirements Document Generator (Article Brief)
# -----------------------------------------------------------------------------
class RequirementsDocumentGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, interviews: list[Interview]) -> str:
        interview_block = "\n".join(
            f"▼ Persona: {iv.persona.name} ({iv.persona.background})\n   Q: {iv.question}\n   A: {iv.answer}"
            for iv in interviews
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a content strategist who turns research into a clear article brief."
            ),
            (
                "human",
                "Using the [Article Topic] and [Interview Findings] below, create an **Article Instruction Document** in English.\n"
                "Include these sections:\n"
                "1. Article Purpose\n"
                "2. Target Readers (demographics, knowledge level, motivations)\n"
                "3. Readers' Pain Points & Needs\n"
                "4. SEO Target Keywords & Related Topics\n"
                "5. Article Outline (headings & sub‑headings)\n"
                "6. Tone & Style Guidelines (voice, reading level, formality)\n"
                "7. Additional Notes / References\n\n"
                "[Article Topic]\n{user_request}\n\n"
                "[Interview Findings]\n{interview_block}"
            ),
        ])

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"user_request": user_request, "interview_block": interview_block})

# -----------------------------------------------------------------------------
# 5. Article Documentation Agent (LangGraph)
# -----------------------------------------------------------------------------
class ArticleDocumentationAgent:
    def __init__(self, llm: ChatOpenAI, k: Optional[int] = 3):
        self.persona_generator = PersonaGenerator(llm, k)
        self.interview_conductor = InterviewConductor(llm)
        self.requirements_generator = RequirementsDocumentGenerator(llm)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        g = StateGraph(ArticleInterviewState)
        g.add_node("generate_personas", self._generate_personas)
        g.add_node("conduct_interviews", self._conduct_interviews)
        g.add_node("generate_requirements", self._generate_requirements)
        g.set_entry_point("generate_personas")
        g.add_edge("generate_personas", "conduct_interviews")
        g.add_edge("conduct_interviews", "generate_requirements")
        g.add_edge("generate_requirements", END)
        return g.compile()

    def _generate_personas(self, state: ArticleInterviewState) -> dict[str, Any]:
        new_personas = self.persona_generator.run(state.user_request).personas
        return {"personas": state.personas + new_personas, "iteration": state.iteration + 1}

    def _conduct_interviews(self, state: ArticleInterviewState) -> dict[str, Any]:
        recent_personas = state.personas[-5:]  # limit batch size
        iv_res = self.interview_conductor.run(state.user_request, recent_personas)
        return {"interviews": state.interviews + iv_res.interviews}

    def _generate_requirements(self, state: ArticleInterviewState) -> dict[str, Any]:
        doc = self.requirements_generator.run(state.user_request, state.interviews)
        return {"requirements_doc": doc}

    def run(self, user_request: str) -> str:
        initial = ArticleInterviewState(user_request=user_request)
        final = self.graph.invoke(initial)
        return final["requirements_doc"]

# -----------------------------------------------------------------------------
# 6. Streamlit App (HF Spaces)
# -----------------------------------------------------------------------------

def main():
    st.title("Article Instruction Agent ✏️")

    llm = ChatOpenAI(model_name="gpt-4o-2024-11-20", temperature=0.0)
    agent = ArticleDocumentationAgent(llm, k=3)

    # --- User Inputs
    col1, col2 = st.columns(2)
    with col1:
        article_topic = st.text_input("1️⃣ Article topic / working title:")
        target_length = st.text_input("2️⃣ Desired word count (e.g., 1200):")
        tone = st.selectbox("3️⃣ Preferred tone / style:", ["Neutral", "Friendly", "Professional", "Casual"])
    with col2:
        primary_goal = st.selectbox("4️⃣ Primary goal:", ["Educate", "Convert", "Entertain", "Inspire"])
        audience_type = st.selectbox("5️⃣ Main audience type:", ["Customers", "Peers", "Decision makers", "Students"])
        knowledge_level = st.selectbox("6️⃣ Audience knowledge level:", ["Beginner", "Intermediate", "Advanced"])

    if st.button("Generate Article Brief"):
        with st.spinner("⏳ Generating... this may take ~40 seconds"):
            user_request = (
                f"[Article Topic]: {article_topic}\n"
                f"[Desired Length]: {target_length}\n"
                f"[Tone]: {tone}\n"
                f"[Primary Goal]: {primary_goal}\n"
                f"[Audience]: {audience_type}\n"
                f"[Knowledge Level]: {knowledge_level}"
            )
            brief = agent.run(user_request)
        st.subheader("📝 Generated Article Instruction Document")
        st.write(brief)

if __name__ == "__main__":
    main()
