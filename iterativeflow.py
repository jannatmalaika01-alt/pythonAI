from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

#X platform tweet generator, real and funny

#i will take 3 separate llms
generator_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

evaluator_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

optimizer_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

#state
class TweetState(TypedDict):
    topic:str
    tweet:str
    evaluation:Literal["approved","not_approved"]
    feedback:str
    iteration:int
    max_iteration:int

def generate_tweet(state:TweetState):
    #prompt..send to generator llm .. return the response
    messages=[
        SystemMessage(content="youre a funny and clever X influencer"),
        HumanMessage(content=f'generate a tweet for X on topic {state["topic"]}, it should not exceed 150 characters, should be funny and human like, should not be question answer like, use simple english, should be meme like, friendly. This is version {state["iteration"]+1}')
    ]
    response=generator_llm.invoke(messages).content
    return {"tweet":str(response)}

def evaluate_tweet(state:TweetState):
    #prompt..send generated tweet to evaluator llm
    messages=[
        SystemMessage(content="youre a strict tweet evaluator. Evaluate the tweet for humor, human-like feel, engagement, simplicity and whether it is under 150 characters. Return exactly APPROVED if it is good, otherwise return exactly NOT_APPROVED followed by short feedback."),
        HumanMessage(content=f'evaluate this tweet: "{state["tweet"]}"')
    ]
    response=str(evaluator_llm.invoke(messages).content)

    if "NOT_APPROVED" in response.upper():
        return {"evaluation":"not_approved","feedback":response}
    else:
        return {"evaluation":"approved","feedback":response}

def optimize_tweet(state:TweetState):
    #prompt..send tweet and feedback to optimizer llm
    messages=[
        SystemMessage(content="youre an expert X tweet optimizer. Improve the tweet using the feedback. Keep it funny, human like, meme like, friendly, simple english and under 150 characters."),
        HumanMessage(content=f'original tweet: "{state["tweet"]}"\nevaluator feedback: {state["feedback"]}')
    ]
    response=optimizer_llm.invoke(messages).content
    return {"tweet":str(response),"iteration":state["iteration"]+1}

def route_tweet(state:TweetState):
    if state["evaluation"]=="approved":
        return "approved"
    if state["iteration"]>=state["max_iteration"]:
        return "approved"
    return "optimize"

#create graph
graph=StateGraph(TweetState)

#add nodes
graph.add_node("generate_tweet",generate_tweet)
graph.add_node("evaluate_tweet",evaluate_tweet)
graph.add_node("optimize_tweet",optimize_tweet)

#add edges
graph.add_edge(START,"generate_tweet")
graph.add_edge("generate_tweet","evaluate_tweet")

graph.add_conditional_edges(
    "evaluate_tweet",
    route_tweet,
    {
        "approved":END,
        "optimize":"optimize_tweet"
    }
)

graph.add_edge("optimize_tweet","evaluate_tweet")

#compile
app=graph.compile()

#run
result=app.invoke({
    "topic":"student life",
    "tweet":"",
    "evaluation":"not_approved",
    "feedback":"",
    "iteration":0,
    "max_iteration":3
})

print("Final Tweet:",result["tweet"])
print("Evaluation:",result["evaluation"])
print("Feedback:",result["feedback"])
print("Iterations:",result["iteration"])