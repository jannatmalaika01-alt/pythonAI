from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import operator


# =========================
# GEMINI
# =========================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


# =========================
# STRUCTURED OUTPUT
# =========================

class evaluation_schema(BaseModel):
    feedback: str = Field(description="Detailed feedback of the essay")
    score: int = Field(description="Score of the essay out of 10")


structured_model = llm.with_structured_output(evaluation_schema)


# =========================
# STATE
# =========================

class essayState(TypedDict):
    essay: str

    languagefeedback: str
    analysis_feedback: str
    clarity_feedback: str

    overall_feedback: str

    individual_scores: Annotated[list[int], operator.add]

    avg_score: float

essay =""" Artificial intelligence (AI) transforms education by tailoring lessons to each student, helping teachers save time, and making learning easier for everyone.Personalized LearningAI changes how students study every day.It looks at a student's weak points and strong points.It adjusts quiz difficulty in real time to match skill levels.Students can learn at their own speed instead of following a strict class pace.Helping TeachersTeachers have a lot of daily tasks. AI helps reduce this heavy workload.It automates routine grading for tests and quizzes.It tracks class progress and spots struggling students early.This gives teachers more time to mentor students face-to-face.Improving AccessibilityAI makes schools more inclusive for everyone.Text-to-speech tools help students with visual or hearing issues.Special apps can spot early signs of reading blocks like dyslexia.Language tools translate text instantly for international learners."""
# =========================
# EVALUATION NODES
# =========================

def evaluate_language(state: essayState):

    prompt = f"""
    Evaluate the language quality of the following essay.
    Give detailed feedback and a score out of 10.

    Essay:
    {state['essay']}
    """

    output = structured_model.invoke(prompt)

    return {
        "languagefeedback": output.feedback,
        "individual_scores": [output.score]
    }


def evaluate_analysis(state: essayState):

    prompt = f"""
    Evaluate the analysis and ideas of the following essay.
    Give detailed feedback and a score out of 10.

    Essay:
    {state['essay']}
    """

    output = structured_model.invoke(prompt)

    return {
        "analysis_feedback": output.feedback,
        "individual_scores": [output.score]
    }


def evaluate_thought(state: essayState):

    prompt = f"""
    Evaluate the clarity and organization of thoughts in the following essay.
    Give detailed feedback and a score out of 10.

    Essay:
    {state['essay']}
    """

    output = structured_model.invoke(prompt)

    return {
        "clarity_feedback": output.feedback,
        "individual_scores": [output.score]
    }


# =========================
# FINAL EVALUATION
# =========================

def final_evaluation(state: essayState):

    scores = state["individual_scores"]

    average = sum(scores) / len(scores)

    overall = f"""
    Language Feedback:
    {state['languagefeedback']}

    Analysis Feedback:
    {state['analysis_feedback']}

    Clarity Feedback:
    {state['clarity_feedback']}
    """

    return {
        "overall_feedback": overall,
        "avg_score": average
    }


# =========================
# GRAPH
# =========================

graph = StateGraph(essayState)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)


# PARALLEL
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")


# All three go to final evaluation
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")

graph.add_edge("final_evaluation", END)


# =========================
# COMPILE
# =========================

app = graph.compile()


# =========================
# RUN
# =========================

result = app.invoke({
    "essay": essay,
    "languagefeedback": "",
    "analysis_feedback": "",
    "clarity_feedback": "",
    "overall_feedback": "",
    "individual_scores": [],
    "avg_score": 0
})


print("Average Score:", result["avg_score"])
print("\nOverall Feedback:")
print(result["overall_feedback"])