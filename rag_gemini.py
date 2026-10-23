import os
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# 1. Load environment variables
load_dotenv()


# 2. Read the knowledge-base document
with open("External_KB_for_RAG.txt", "r", encoding="utf-8") as file:
    text = file.read()


# 3. Split the document into chunks
def split_into_chunks(text, chunk_size=300, overlap=50):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


chunks = split_into_chunks(text)

print("Number of chunks:", len(chunks))


# 4. Create embeddings for the chunks
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

embeddings = np.array(embeddings, dtype="float32")


# 5. Create the FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# 6. Create the user's question
question = "Hi. I’m a citizen of Romania. Do I need a Schengen visa if I travel to France?"


# 7. Convert the question into an embedding
question_embedding = model.encode([question])

question_embedding = np.array(
    question_embedding,
    dtype="float32"
)


# 8. Search FAISS for the 5 most relevant chunks
distances, indices = index.search(
    question_embedding,
    k=5
)


# 9. Display the retrieved chunks
print("\nQuestion:")
print(question)

print("\nRetrieved chunks:")

for rank, chunk_index in enumerate(indices[0]):
    print(f"\n--- Retrieved Chunk {rank + 1} ---")
    print("Chunk index:", chunk_index)
    print("Distance:", distances[0][rank])
    print(chunks[chunk_index])
# 10. Combine the retrieved chunks into one context
retrieved_chunks = []

for chunk_index in indices[0]:
    retrieved_chunks.append(chunks[chunk_index])

context = "\n\n".join(retrieved_chunks)
# 12. Create a prompt containing the question and context
prompt = f"""
You are a helpful travel assistant for GoTravel.

Answer the user's question using the context provided below.
If the context does not contain enough information, say that you do not
have enough information to answer confidently.
Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""


# 13. Send the prompt to Gemini
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=prompt
)


# 14. Display the RAG answer
print("\n========== GEMINI RAG ANSWER ==========")
print(response.text)