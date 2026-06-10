"""D1 spike — LangChain + LangGraph implementation.

Run:  uv run --group spike python spikes/d1/spike_langgraph.py
Same pipeline as spike_llamaindex.py: load 2 books -> token splitter (500/50)
-> embed locally (qwen3-embedding-0.6b, query prefix policy) -> in-memory
vector store (persisted to .cache/) -> retrieve top-6 -> streamed, citation-
forced answer from local gemma-12b, orchestrated as a LangGraph state graph.
"""

import sys
import time
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    ANSWER_PROMPT,
    CHAT_MODEL,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    EMBED_MODEL,
    LLAMA_SWAP_BASE,
    QUESTIONS,
    QWEN3_QUERY_PREFIX,
    TOP_K,
    format_context,
    load_books,
)
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

CACHE_FILE = Path(__file__).parent / ".cache" / "langgraph-store.json"


class Qwen3Embeddings(OpenAIEmbeddings):
    """The 'one embedding module owning the prefix policy' rule, LangChain flavor:
    queries get the Qwen3 instruction prefix, documents don't."""

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(QWEN3_QUERY_PREFIX + text)

    async def aembed_query(self, text: str) -> list[float]:
        return await super().aembed_query(QWEN3_QUERY_PREFIX + text)


def build_or_load_store(embeddings: Qwen3Embeddings) -> InMemoryVectorStore:
    if CACHE_FILE.exists():
        print("Loading persisted store...")
        return InMemoryVectorStore.load(str(CACHE_FILE), embeddings)

    print("Building store (chunk + embed)...")
    t0 = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )
    docs: list[Document] = []
    for book in load_books():
        for chunk in splitter.split_text(book.text):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"author": book.author, "title": book.title},
                )
            )
    store = InMemoryVectorStore(embeddings)
    store.add_documents(docs)
    print(f"Store built ({len(docs)} chunks) in {time.perf_counter() - t0:.1f}s")
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    store.dump(str(CACHE_FILE))
    return store


class RagState(TypedDict, total=False):
    question: str
    passages: list[tuple[str, str]]
    scores: list[float]
    answer: str


def main() -> None:
    embeddings = Qwen3Embeddings(
        model=EMBED_MODEL,
        base_url=LLAMA_SWAP_BASE,
        api_key="none",
        check_embedding_ctx_length=False,  # required for non-OpenAI endpoints
        chunk_size=16,
    )
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        base_url=LLAMA_SWAP_BASE,
        api_key="none",
        temperature=0.2,
    )
    store = build_or_load_store(embeddings)

    def retrieve(state: RagState) -> dict[str, object]:
        results = store.similarity_search_with_score(state["question"], k=TOP_K)
        passages = [
            (f"{doc.metadata.get('author')}, {doc.metadata.get('title')}", doc.page_content)
            for doc, _score in results
        ]
        return {"passages": passages, "scores": [score for _doc, score in results]}

    def generate(state: RagState) -> dict[str, object]:
        prompt = ANSWER_PROMPT.format(
            context=format_context(state["passages"]), question=state["question"]
        )
        # Node invokes the LLM; token streaming surfaces via graph stream_mode="messages".
        response = llm.invoke(prompt)
        return {"answer": response.content}

    graph = (
        StateGraph(RagState)
        .add_node("retrieve", retrieve)
        .add_node("generate", generate)
        .add_edge(START, "retrieve")
        .add_edge("retrieve", "generate")
        .add_edge("generate", END)
        .compile()
    )

    for question in QUESTIONS:
        print(f"\n{'=' * 78}\nQ: {question}\n")
        t0 = time.perf_counter()
        shown_retrieval = False
        first_token: float | None = None
        for item, meta in graph.stream({"question": question}, stream_mode=["updates", "messages"]):
            if item == "updates":
                update = meta
                if "retrieve" in update and not shown_retrieval:
                    shown_retrieval = True
                    passages = update["retrieve"]["passages"]
                    scores = update["retrieve"]["scores"]
                    for i, ((label, text), score) in enumerate(
                        zip(passages, scores, strict=True), start=1
                    ):
                        preview = text[:90].replace("\n", " ")
                        print(f"  [{i}] score={score:.3f} {label}: {preview}...")
                    print("\nA: ", end="", flush=True)
            elif item == "messages":
                token_chunk, _meta = meta
                if token_chunk.content:
                    if first_token is None:
                        first_token = time.perf_counter() - t0
                    print(token_chunk.content, end="", flush=True)
        print(f"\n\n  first-token={first_token or 0:.1f}s  total={time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
