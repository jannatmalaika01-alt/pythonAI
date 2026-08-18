from google import genai

client = genai.Client()

def ask_ai(prompt):
    """Shared function — every menu option calls this instead of repeating the API call."""
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config={
            "system_instruction": "You are a concise study assistant.",
            "thinking_config": {"thinking_level": "high"},
        }
    )
    return response.text

print("Starting the AI model generation...")
print("Select an option:")
print(" 1) Explain a topic")
print(" 2) Summarize a topic")
print(" 3) Generate MCQ questions for a topic")

choice = input("Enter your choice (1, 2, or 3): ")

if choice == "1":
    topic = input("Enter your topic to explain: ")
    print("AI:")
    print(ask_ai(f"Explain {topic} in simple terms in 3 sentences."))
elif choice == "2":
    topic = input("Enter your topic to summarize: ")
    print("AI:")
    print(ask_ai(f"Summarize {topic} in 3 sentences."))
elif choice == "3":
    topic = input("Enter your topic to generate MCQ questions for: ")
    print("AI:")
    print(ask_ai(f"Generate 3 multiple choice questions with answers for {topic}."))
else:
    print("Invalid choice — please enter 1, 2, or 3.")