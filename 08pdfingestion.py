import chromadb
from google import genai
from pypdf import PdfReader

client_genai = genai.Client()
client_chroma = chromadb.PersistentClient(path="./study_buddy_db")
# NEW collection name so this doesn't mix with your earlier hardcoded test data
collection = client_chroma.get_or_create_collection(name="real_notes")

def get_embedding(text):
    result = client_genai.models.embed_content(model="gemini-embedding-001", contents=text)
    return result.embeddings[0].values

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks of chunk_size characters, with a bit of overlap
    so we don't cut a sentence in half right at a chunk boundary and lose meaning."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # step forward, but re-include the last bit
    return [c.strip() for c in chunks if c.strip()]  # drop empty/whitespace-only chunks

# --- Step 1-3: ingest the PDF ---
pdf_path = "web servlets.pdf"  # put your actual PDF filename here, same folder as this script
raw_text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(raw_text)

print(f"Extracted {len(raw_text)} characters, split into {len(chunks)} chunks.")

# --- Step 4-5: embed + store (same pattern as before) ---
if collection.count() == 0:
    embeddings = [get_embedding(c) for c in chunks]
    collection.add(documents=chunks, embeddings=embeddings, ids=[f"chunk_{i}" for i in range(len(chunks))])
    print(f"Stored {len(chunks)} chunks in Chroma.")
else:
    print(f"Collection already has {collection.count()} chunks — skipping re-embedding.")

# --- Now ask a real question about YOUR PDF ---
question = input("\nAsk a question about your notes: ")
question_embedding = get_embedding(question)
results = collection.query(query_embeddings=[question_embedding], n_results=3)
context = "\n".join(f"- {c}" for c in results["documents"][0])

rag_prompt = f"""Answer using ONLY the context below. If it's not there, say so.

Context:
{context}

Question: {question}
Answer:"""

response = client_genai.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=rag_prompt,
    config={"system_instruction": "You are a study assistant that only answers from provided notes."}
)
print("\nAI:", response.text)