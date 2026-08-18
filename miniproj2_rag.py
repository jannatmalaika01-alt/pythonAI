import chromadb
from google import genai

client_genai = genai.Client()
client_chroma = chromadb.PersistentClient(path="./study_buddy_db")
collection = client_chroma.get_or_create_collection(name="student_notes")

notes_chunks = [
    "The mitochondria is the powerhouse of the cell, producing ATP through cellular respiration.",
    "Photosynthesis occurs in chloroplasts and converts sunlight into chemical energy stored as glucose.",
    "The French Revolution began in 1789 due to financial crisis and social inequality.",
    "Newton's second law states that force equals mass times acceleration (F=ma).",
]

def get_embedding(text):
    result = client_genai.models.embed_content(model="gemini-embedding-001", contents=text)
    return result.embeddings[0].values

if collection.count() == 0:
    embeddings = [get_embedding(chunk) for chunk in notes_chunks]
    collection.add(documents=notes_chunks, embeddings=embeddings, ids=[f"chunk_{i}" for i in range(len(notes_chunks))])

# --- Everything above this line is identical to file 05 ---
# --- Everything below is NEW: step 5, closing the loop ---

question = "why do plants need sunlight?"
question_embedding = get_embedding(question)

results = collection.query(query_embeddings=[question_embedding], n_results=2)
retrieved_chunks = results["documents"][0]  # just the text, ignore distances for this step

# Build context block from retrieved chunks
context = "\n".join(f"- {chunk}" for chunk in retrieved_chunks)

# This is the actual RAG prompt — instructing the model to stick to the given context
rag_prompt = f"""Answer the question using ONLY the context below. 
If the context doesn't contain the answer, say "I don't have that information in your notes."

Context:
{context}

Question: {question}

Answer:"""

response = client_genai.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=rag_prompt,
    config={"system_instruction": "You are a study assistant that only answers from provided notes."}
)

print("Retrieved context:")
print(context)
print("\nAI answer (grounded in retrieved notes):")
print(response.text)