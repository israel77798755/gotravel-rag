with open("External_KB_for_RAG.txt", "r", encoding="utf-8") as file:
    text = file.read()


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

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)
chunks = split_into_chunks(text)
print("First chunk length:", len(chunks[0]))