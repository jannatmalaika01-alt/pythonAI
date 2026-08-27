# start -- generate outline -- generate blog -- end
from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)
# states define topic,outline,blog content all str
class llm_state(TypedDict):
    topic:str
    outline:str
    blog:str
def create_outline(state:llm_state)->llm_state:
    topic=state["topic"]
    prompt=f"Create an outline for a blog post about {topic}"
    outline=llm.invoke(prompt).content
    state["outline"]=outline
    return state
def create_blog(state:llm_state)->llm_state:
    outline=state["outline"]
    prompt=f"Create a blog post based on the following outline: {outline}"
    blog=llm.invoke(prompt).content
    state["blog"]=blog
    return state
#create graph
graph=StateGraph(llm_state)
#add nodes
graph.add_node("create_outline", create_outline)
graph.add_node("create_blog", create_blog)
#add edges
graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "create_blog")
graph.add_edge("create_blog", END)
#compile
workflow=graph.compile()
#execute
result=workflow.invoke({
    "topic":"The future of AI",
    "outline":"",
    "blog":""
})
print(result)