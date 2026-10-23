
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Read the knowledge base
with open("External_KB_for_RAG.txt", "r", encoding="utf-8") as file:
    text = file.read()


# 2. Split the document into chunks
def split_into_chunks(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


chunks = split_into_chunks(text)


# 3. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 4. Create embeddings for all chunks
embeddings = model.encode(chunks)

# FAISS expects float32 vectors
embeddings = np.array(embeddings, dtype="float32")


# 5. Create a FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Store the chunk embeddings
index.add(embeddings)


# 6. Ask a question
question = "I am a Romanian citizen. Do I need a Schengen visa to travel to France?"

# Convert the question into an embedding
question_embedding = model.encode([question])
question_embedding = np.array(question_embedding, dtype="float32")


# 7. Retrieve the 5 nearest chunks
distances, indices = index.search(question_embedding, k=5)


# 8. Display the retrieved chunks
print("Question:", question)

for rank, chunk_index in enumerate(indices[0]):
    print(f"\n--- Retrieved Chunk {rank + 1} ---")
    print("Chunk index:", chunk_index)
    print("Distance:", distances[0][rank])
    print(chunks[chunk_index])
