from langgraph.graph import StateGraph,START,END
from typing import TypedDict, Literal 
class QuadState(TypedDict):
    a:int
    b:int
    c:int
    equation:str
    discriminant:float
    result:str
def show_eq(state:QuadState):
    equation=f"{state['a']}x2{state['b']}x{state['c']}" 
    return {"equation": equation}
def calc_disc(state:QuadState):
    #b**2 - 4*a*c
    discriminant=state['b']**2 - 4*state['a']* state['c']
    return {"discriminant": discriminant}
def real_roots(state:QuadState):
    #-b  + sq disc /2a
   root1= (-state["b"] + state["discriminant"]**0.5) / (2 * state["a"])
   root2= (-state["b"] - state["discriminant"]**0.5) /(2 * state["a"])
   result=f'the roots are {root1} and {root2}'
   return {'result':result}
def repeated_roots(state:QuadState):
    #-b  + sq disc /2a
   root= (-state["b"] ) / (2 * state["a"])
   result=f'the only repeating roots is {root}'
   return {'result':result}
def no_real_roots(state:QuadState):

   result=f'there are no real roots'
   return {'result':result}
def check_condition(state:QuadState)->Literal["real_roots","no_real_roots","repeated_roots"]:
    if state["discriminant"]>0:
        return "real_roots"
    elif state["discriminant"]==0:
        return "repeated_roots"
    else:
        return "no_real_roots"

graph= StateGraph(QuadState)
graph.add_node("show_eq",show_eq)
graph.add_node("calc_disc",calc_disc)
graph.add_node("real_roots",real_roots)
graph.add_node("no_real_roots",no_real_roots)
graph.add_node("repeated_roots",repeated_roots)
graph.add_edge(START,"show_eq")
graph.add_edge("show_eq","calc_disc")

graph.add_conditional_edges("calc_disc",check_condition)

graph.add_edge("real_roots",END)
graph.add_edge("no_real_roots",END)
graph.add_edge("repeated_roots",END)
workflow=graph.compile()
initial_state={
    "a":4,
    "b":-5,
    "c":-4
}
result=workflow.invoke(initial_state)
print(result["result"])