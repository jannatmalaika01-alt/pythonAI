from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class BMIState(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float
    category:str

def calculate_bmi(state: BMIState) -> BMIState:

    weight = state["weight_kg"]
    height = state["height_m"]

    bmi = weight / (height * height)

    return {"bmi": round(bmi, 2)}

def categorize_bmi(state: BMIState) -> BMIState:
    bmi = state["bmi"]

    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        category = "Normal weight"
    elif 25 <= bmi < 29.9:
        category = "Overweight"
    else:
        category = "Obesity"

    return {"category": category}
# Define graph
graph = StateGraph(BMIState)

# Add node
graph.add_node("calculate_bmi", calculate_bmi)
graph.add_node("categorize_bmi", categorize_bmi)
# Add edges
graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", "categorize_bmi")
graph.add_edge("categorize_bmi", END)

# Compile
workflow = graph.compile()


# Execute graph
initial_state = {
    "weight_kg": 70.0,
    "height_m": 1.75
}

final_state = workflow.invoke(initial_state)

print(final_state)