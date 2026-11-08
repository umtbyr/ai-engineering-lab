from contextlib import asynccontextmanager
from pathlib import Path
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer
from uuid import UUID, uuid4
from app.db.database import create_pool
from app.rag.ingestion import ingest_markdown
from app.rag.pg_retrieval import PostgresChunkSearcher, HybridRetriever
from app.rag.rag import RAGAssistant
from app.rag.reranking import RerankingRetriever
from app.agent.agent import create_engineering_agent
from langchain.mcp import MCPAdapter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
# -----------------------------
# API Request / Response Models
# -----------------------------


class AgentStep(BaseModel):
    type: str
    content: str
    tool_calls: list[dict] = Field(default_factory=list)


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep]


class AgentChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: UUID | None = None


class AgentResponse(BaseModel):
    conversation_id: UUID
    answer: str
    steps: list[AgentStep]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SourceResponse(BaseModel):
    citation: str
    chunk_id: str
    source: str | None
    heading_path: list[str]
    source_lines: list[list[int]]
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class DocumentRequest(BaseModel):
    source: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=200_000)


# -----------------------------
# Application Lifecycle
# -----------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing RAG engine...")

    # 1. Embedding modeli
    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 2. Reranking modeli
    reranker_model = CrossEncoder(
        "BAAI/bge-reranker-v2-m3",
        device="cpu",
    )

    # in memory checkpointer
    checkpointer = InMemorySaver()

    # 3. Database connection pool
    pool = create_pool()

    try:
        pool.open(wait=True, timeout=10)

        # 4. Retrieval pipeline
        searcher = PostgresChunkSearcher(
            model=embedding_model,
            pool=pool,
        )

        hybrid_retriever = HybridRetriever(
            searcher=searcher,
            candidate_k=20,
        )

        reranking_retriever = RerankingRetriever(
            retriever=hybrid_retriever,
            reranker=reranker_model,
            candidate_k=10,
        )

        # 5. Chat modeli
        chat_model = ChatOllama(
            model="qwen3:4b-instruct",
            temperature=0,
            num_ctx=16384,
        )

        # 6. Ortak servisler
        app.state.embedding_model = embedding_model
        app.state.db_pool = pool

        app.state.rag = RAGAssistant(
            retriever=reranking_retriever,
            model=chat_model,
        )

        mcp_config = {
            "mcpServers": {
                "playwright": {
                    "command": "cmd",
                    "args": [
                        "/c",
                        "npx",
                        "--yes",
                        "@playwright/mcp@latest",
                        "--headless",
                        "--isolated",
                        "--no-webmcp",
                    ],
                }
            }
        }

        async with MCPAdapter(mcp_config) as adapter:
            available_tools = await adapter.list_tools()

            allowed_tools = {
                "browser_navigate",
                "browser_snapshot",
                "browser_click",
            }

            browser_tools = [
                tool for tool in available_tools if tool.name in allowed_tools
            ]

            print("Agent browser tools:", [tool.name for tool in browser_tools])
            app.state.agent = create_engineering_agent(
                retriever=reranking_retriever,
                model=chat_model,
                checkpointer=checkpointer,
                browser_tools=browser_tools,
            )

            print("Engineering Agent ready.")

        yield

    finally:
        pool.close()


# -----------------------------
# FastAPI Application
# -----------------------------

app = FastAPI(
    title="RAG Knowledge Assistant",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/db-health")
def db_health(request: Request):

    pool = request.app.state.db_pool

    with pool.connection() as conn:
        result = conn.execute("""
            SELECT
                current_database() AS database,
                pg_backend_pid() AS connection_pid
            """).fetchone()

    return {
        "status": "connected",
        "database": result["database"],
        "connection_pid": result["connection_pid"],
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request):

    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty",
        )

    rag: RAGAssistant = request.app.state.rag

    result = rag.ask(question, top_k=3)

    return ChatResponse(
        answer=result.text,
        sources=[
            SourceResponse(
                citation=source.citation,
                chunk_id=source.chunk_id,
                source=source.source,
                heading_path=list(source.heading_path),
                source_lines=[list(pair) for pair in source.source_lines],
                score=round(source.score, 4),
            )
            for source in result.sources
        ],
    )


@app.post("/api/agent/chat", response_model=AgentResponse)
async def agent_chat(
    payload: AgentChatRequest,
    request: Request,
):
    question = payload.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty",
        )

    conversation_id = payload.conversation_id or uuid4()

    config = {
        "configurable": {"thread_id": str(conversation_id)},
        "recursion_limit": 20,
    }

    agent = request.app.state.agent
    async for update in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="updates",
    ):
        for node_name, node_output in update.items():
            print(f"[LangGraph] Node executed: {node_name}")

            for message in node_output.get("messages", []):
                print(f"  Message: {type(message).__name__}")

                for call in getattr(message, "tool_calls", []):
                    print(f"  Tool requested: {call['name']}")

    snapshot = await agent.aget_state(config)
    messages = snapshot.values["messages"]

    steps = [
        AgentStep(
            type=type(message).__name__,
            content=message.text,
            tool_calls=getattr(message, "tool_calls", []),
        )
        for message in messages
    ]

    return AgentResponse(
        conversation_id=conversation_id,
        answer=messages[-1].text,
        steps=steps,
    )


@app.post("/api/documents")
def add_document(
    payload: DocumentRequest,
    request: Request,
):
    if not payload.source.strip() or not payload.content.strip():
        raise HTTPException(
            status_code=422,
            detail="Source and content cannot be blank",
        )

    try:
        return ingest_markdown(
            source=payload.source,
            content=payload.content,
            model=request.app.state.embedding_model,
            pool=request.app.state.db_pool,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
