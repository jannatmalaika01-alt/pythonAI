from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# =========================
# SENTIMENT SCHEMA
# =========================

class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="sentiment of the review"
    )


structured_model = llm.with_structured_output(SentimentSchema)


# =========================
# DIAGNOSIS SCHEMA
# =========================

class DiagnosisSchema(BaseModel):
    issue_type: Literal[
        "ux", "ui", "software", "hardware", "others"
    ] = Field(description="the type of issue")

    tone: Literal[
        "angry", "frustrated", "calm", "disappointed"
    ] = Field(description="the type of tone")

    urgency: Literal[
        "high", "moderate", "low"
    ] = Field(description="how urgent the issue is")


structured_model2 = llm.with_structured_output(DiagnosisSchema)


# =========================
# STATE
# =========================

class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str


# =========================
# FIND SENTIMENT
# =========================

def find_sentiment(state: ReviewState):

    prompt = f"""
    For the following review, find the sentiment:

    {state["review"]}
    """

    sentiment = structured_model.invoke(prompt).sentiment

    return {"sentiment": sentiment}


# =========================
# CHECK SENTIMENT
# =========================

def check_sentiment(state: ReviewState) -> Literal[
    "positive_response",
    "run_diagnosis"
]:

    if state["sentiment"] == "positive":
        return "positive_response"

    else:
        return "run_diagnosis"


# =========================
# POSITIVE RESPONSE
# =========================

def positive_response(state: ReviewState):

    prompt = f"""
    Write a warm thank-you message for this positive customer review:

    {state["review"]}
    """

    response = llm.invoke(prompt).content

    return {"response": response}


# =========================
# RUN DIAGNOSIS
# =========================

def run_diagnosis(state: ReviewState):

    prompt = f"""
    Diagnose this negative customer review.

    Return:
    - issue_type
    - tone
    - urgency

    Review:
    {state["review"]}
    """

    response = structured_model2.invoke(prompt)

    return {"diagnosis": response.model_dump()}


# =========================
# NEGATIVE RESPONSE
# =========================

def negative_response(state: ReviewState):

    prompt = f"""
    Write a polite and empathetic response to this negative customer review.

    Review:
    {state["review"]}

    Diagnosis:
    {state["diagnosis"]}

    Acknowledge the customer's issue and offer help.
    """

    response = llm.invoke(prompt).content

    return {"response": response}


# =========================
# GRAPH
# =========================

graph = StateGraph(ReviewState)

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)


# START → sentiment
graph.add_edge(START, "find_sentiment")


# sentiment → positive OR diagnosis
graph.add_conditional_edges(
    "find_sentiment",
    check_sentiment
)


# positive → END
graph.add_edge("positive_response", END)


# diagnosis → negative response
graph.add_edge("run_diagnosis", "negative_response")


# negative response → END
graph.add_edge("negative_response", END)


# =========================
# COMPILE
# =========================

workflow = graph.compile()


# =========================
# TEST
# =========================

initial_state = {
    "review": "The product was really bad and keeps crashing. I am very frustrated."
}

result = workflow.invoke(initial_state)

print("\nFINAL RESULT:\n")
print(result)
print("\nSENTIMENT:", result["sentiment"])
print("\nDIAGNOSIS:", result.get("diagnosis"))
print("\nRESPONSE:", result["response"])