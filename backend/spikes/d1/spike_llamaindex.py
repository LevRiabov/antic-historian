"""D1 spike — LlamaIndex implementation.

Run:  uv run --group spike python spikes/d1/spike_llamaindex.py
Pipeline: load 2 books -> sentence-split (500/50 tokens) -> embed locally
(qwen3-embedding-0.6b, query prefix policy) -> in-memory vector index
(persisted to .cache/) -> retrieve top-6 -> streamed, citation-forced answer
from local gemma-12b.
"""

import sys
import time
from pathlib import Path

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
from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import ChatMessage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike

PERSIST_DIR = Path(__file__).parent / ".cache" / "llamaindex"


class Qwen3Embedding(OpenAILikeEmbedding):
    """The 'one embedding module owning the prefix policy' rule, LlamaIndex flavor:
    override query-side embedding to apply the Qwen3 instruction prefix."""

    def _get_query_embedding(self, query: str) -> list[float]:
        return super()._get_query_embedding(QWEN3_QUERY_PREFIX + query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await super()._aget_query_embedding(QWEN3_QUERY_PREFIX + query)


def build_or_load_index(embed_model: Qwen3Embedding) -> VectorStoreIndex:
    if PERSIST_DIR.exists():
        print("Loading persisted index...")
        storage = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
        index = load_index_from_storage(storage, embed_model=embed_model)
        assert isinstance(index, VectorStoreIndex)
        return index

    print("Building index (chunk + embed)...")
    t0 = time.perf_counter()
    docs = [
        Document(
            text=book.text,
            metadata={"author": book.author, "title": book.title},
        )
        for book in load_books()
    ]
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS)
    index = VectorStoreIndex.from_documents(
        docs,
        transformations=[splitter],
        embed_model=embed_model,
        show_progress=True,
    )
    print(f"Index built in {time.perf_counter() - t0:.1f}s")
    index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    return index


def answer_question(question: str, retriever: BaseRetriever, llm: OpenAILike) -> None:
    print(f"\n{'=' * 78}\nQ: {question}\n")
    t0 = time.perf_counter()
    nodes = retriever.retrieve(question)
    t_retrieve = time.perf_counter() - t0

    passages: list[tuple[str, str]] = []
    for node_with_score in nodes:
        node = node_with_score.node
        label = f"{node.metadata.get('author')}, {node.metadata.get('title')}"
        passages.append((label, node.get_content()))
        preview = node.get_content()[:90].replace("\n", " ")
        print(f"  [{len(passages)}] score={node_with_score.score:.3f} {label}: {preview}...")

    prompt = ANSWER_PROMPT.format(context=format_context(passages), question=question)
    print("\nA: ", end="", flush=True)
    t1 = time.perf_counter()
    first_token: float | None = None
    response = llm.stream_chat([ChatMessage(role="user", content=prompt)])
    for chunk in response:
        if first_token is None:
            first_token = time.perf_counter() - t1
        print(chunk.delta, end="", flush=True)
    total = time.perf_counter() - t1
    print(
        f"\n\n  retrieve={t_retrieve * 1000:.0f}ms  "
        f"first-token={first_token or 0:.1f}s  generate={total:.1f}s"
    )


def main() -> None:
    embed_model = Qwen3Embedding(
        model_name=EMBED_MODEL,
        api_base=LLAMA_SWAP_BASE,
        api_key="none",
        embed_batch_size=16,
    )
    llm = OpenAILike(
        model=CHAT_MODEL,
        api_base=LLAMA_SWAP_BASE,
        api_key="none",
        is_chat_model=True,
        context_window=16384,
        temperature=0.2,
    )
    index = build_or_load_index(embed_model)
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    for question in QUESTIONS:
        answer_question(question, retriever, llm)


if __name__ == "__main__":
    main()
