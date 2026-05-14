import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from pathlib import Path
    import chromadb
    from openai import OpenAI

    import httpx

    return OpenAI, Path, chromadb, httpx, mo


@app.cell
def _():
    import os
    print("Ключ найден:" if "OPENAI_API_KEY" in os.environ else "Ключ не найден")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: Text Chunking

    Split documents into smaller chunks with overlap to preserve context.
    """)
    return


@app.function
def chunk_text(text, size=1000, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            bp = text.rfind("\n\n", start, end)
            if bp == -1:
                bp = text.rfind(". ", start, end)
            if bp > start:
                end = bp + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end
    return chunks


@app.cell
def _(Path):
    file_path = Path("../../datasets/text_files/harry_potter_knowledge_base.txt")
    text = file_path.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    return (chunks,)


@app.cell
def _(chunks):
    chunks
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: Embed and Store in ChromaDB

    Generate embeddings and store them in a vector database.
    """)
    return


@app.cell
def _(OpenAI, chromadb, httpx):

    client = OpenAI(http_client=httpx.Client())
    embedding_model = "text-embedding-3-small"


    def embed_and_store(chunks, db_path, collection_name):
        chroma = chromadb.PersistentClient(path=str(db_path))
        collection = chroma.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Harry Potter knowledge base"},
        )

        embeddings = []
        for i in range(0, len(chunks), 100):
            batch = chunks[i : i + 100]
            res = client.embeddings.create(model=embedding_model, input=batch)
            embeddings.extend([x.embedding for x in res.data])

        collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"chunk_index": i} for i in range(len(chunks))],
        )
        return collection

    return client, embed_and_store, embedding_model


@app.cell
def _(Path, chunks, embed_and_store):
    chroma_db_dir = Path("chroma_db")
    collection = embed_and_store(chunks, chroma_db_dir, "harry_potter_kb")
    collection
    return (collection,)


@app.cell
def _(collection):
    print(f"Количество записей в базе: {collection.count()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: Test Retrieval

    Test that the retrieval system finds relevant chunks for a given question.
    """)
    return


@app.cell
def _(client, collection, embedding_model):
    def retrieve_with_scores(question, top_k=3):
        q_emb = client.embeddings.create(
            model=embedding_model,
            input=question,
        ).data[0].embedding

        res = collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "distances"],
        )

        results = []
        for doc, dist in zip(res["documents"][0], res["distances"][0]):
            results.append({
                "content": doc,
                "distance": round(dist, 4) # Чем меньше число, тем ближе смысл
            })

        for r in results:
            print(f"Score: {r['distance']} | Text: {r['content'][:100]}...")

        return res["documents"][0]


    return (retrieve_with_scores,)


@app.cell
def _(retrieve_with_scores):
    question = "Why did Uncle Vernon take the family to a hut in the middle of the sea?"
    results = retrieve_with_scores(question)
    return (results,)


@app.cell
def _(results):
    results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: Test Generation

    Feed the retrieved documents and question to the LLM to generate an answer.
    """)
    return


@app.cell
def _(client):
    def answer(question, docs):
        context = "\n\n---\n\n".join(docs)
        prompt = f"""Answer the question using only the context below.

    Context:
    {context}

    Question:
    {question}

    Answer:"""

        res = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
        )

        return res.choices[0].message.content


    return (answer,)


@app.cell
def _(answer, retrieve_with_scores):
    query = "Why did Uncle Vernon take the family to a hut in the middle of the sea?"
    docs = retrieve_with_scores(query)
    answer_text = answer(query, docs)
    return (answer_text,)


@app.cell
def _(answer_text):
    answer_text
    return


if __name__ == "__main__":
    app.run()
