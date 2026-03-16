"""
AI Foundry V2 RAG with NEW Agent Pattern (create_version + conversations)
=========================================================================

Architecture: AI Foundry V2 (account + project, no Hub)
  Resource type : Microsoft.CognitiveServices/accounts
    Account       : <account-name-from-env>
    Project       : <project-name-from-env>
    Account URL   : <account-endpoint-from-env>
    Project URL   : <project-endpoint-from-env>

Two-phase workflow
------------------
Phase 1 – Index ./data/Joakim.md into Azure AI Search
  • Parses the markdown into per-header chunks
  • Generates vector embeddings with text-embedding-3-large
    (deployed on the AI Foundry account – RBAC, no key)
  • Creates / recreates the search index with HNSW vector support
  • Uploads the chunks + embeddings to AI Search

Phase 2 – Create RAG Agent + Conversation/Responses (NEW Foundry pattern)
    • Creates agent version via create_version() + PromptAgentDefinition
    • Attaches AI Search tool to agent for RAG grounding
    • Uses get_openai_client() to obtain OpenAI inference client
    • Creates conversation for multi-turn context
    • Invokes agent via responses.create() with agent_reference
    • Prints the answer + sources

NEW Foundry Pattern (per Foundry documentation):
    • AIProjectClient + create_version() + PromptAgentDefinition
    • get_openai_client() → conversations.create() → responses.create()
    • agent_reference in extra_body for agent invocation

Authentication: DefaultAzureCredential (RBAC only – no account keys).
    RBAC scope for embeddings (account endpoint): https://cognitiveservices.azure.com/.default
    RBAC scope for project API (project endpoint): https://ai.azure.com/.default

Required packages
-----------------
    pip install azure-ai-projects azure-search-documents azure-identity openai
"""

import argparse
import os
import string
import random
import time
import json
from pathlib import Path
import re

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# AI Foundry project client + new pattern imports
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    PromptAgentDefinition,
    AzureAISearchTool,
    AzureAISearchQueryType,
    AzureAISearchToolResource,
    AISearchIndexResource,
)
from openai import AzureOpenAI
from dotenv import load_dotenv

# AI Search SDK
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
)

# ============================================================================
# Configuration  (sourced from aifactory-usecase-config.yaml)
# ============================================================================

_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(_BASE_DIR.parent / ".env")


def _env(key: str, default: str | None = None) -> str:
    value = os.getenv(key, default)
    return None if value == "<TODO>" else value


def _env_int(key: str, default: str | None = None) -> int | None:
    value = _env(key, default)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {key} must be an integer; got '{value}'") from exc


SUBSCRIPTION_ID        = _env("SUBSCRIPTION_ID", "<TODO>")
TENANT_ID              = _env("TENANT_ID", "<TODO>")
RESOURCE_GROUP         = _env("RESOURCE_GROUP", "<TODO>")

# AI Foundry V2 –  account endpoint  (Microsoft.CognitiveServices/accounts)
# Models are deployed at the account level; no Hub in V2.
FOUNDRY_ACCOUNT_ENDPOINT = _env("FOUNDRY_ACCOUNT_ENDPOINT", "<TODO>")

# AI Foundry V2 – project endpoint  (child of the account)
FOUNDRY_PROJECT_ENDPOINT = _env("FOUNDRY_PROJECT_ENDPOINT", "<TODO>")

# Azure AI Search
AI_SEARCH_NAME         = _env("AI_SEARCH_NAME", "<TODO>")
AI_SEARCH_ENDPOINT     = f"https://{AI_SEARCH_NAME}.search.windows.net"
AI_SEARCH_INDEX_NAME   = _env("AI_SEARCH_INDEX_NAME", "<TODO>")

# Models
EMBEDDING_MODEL        = _env("EMBEDDING_MODEL", "<TODO>")
EMBEDDING_DIMENSIONS   = _env_int("EMBEDDING_DIMENSIONS", "<TODO>")
CHAT_MODEL             = _env("CHAT_MODEL", "<TODO>")

# Agent
AGENT_NAME             = _env("AGENT_NAME", "<TODO>")
AGENT_INSTRUCTIONS     = _env(
    "AGENT_INSTRUCTIONS",
    "<TODO>",
)
FIRST_PROMPT           = _env("FIRST_PROMPT", "<TODO>")
AI_SEARCH_CONNECTION_NAME = _env("AI_SEARCH_CONNECTION_NAME", "Connection-AzureAISearch-Foundry")
AGENT_STATE_FILE       = Path(__file__).with_name(".agent_04_state.json")

# Data file (this script lives in usecase_code/agent_foundry/)
DATA_FILE = Path(__file__).parent.parent / "data" / "Joakim.md"


# ============================================================================
# Phase 1 helpers: markdown parsing, index creation, document upload
# ============================================================================

def parse_markdown_chunks(filepath: Path) -> list[dict]:
    """Split markdown into chunks at every H1 / H2 heading."""
    text = filepath.read_text(encoding="utf-8")
    chunks: list[dict] = []
    current_title = filepath.stem
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## ") or line.startswith("# "):
            # Flush previous chunk
            content = "\n".join(current_lines).strip()
            if content:
                chunks.append(
                    {
                        "id": f"chunk-{len(chunks)}",
                        "title": current_title,
                        "content": content,
                        "source": filepath.name,
                    }
                )
            current_title = line.lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    # Flush final chunk
    content = "\n".join(current_lines).strip()
    if content:
        chunks.append(
            {
                "id": f"chunk-{len(chunks)}",
                "title": current_title,
                "content": content,
                "source": filepath.name,
            }
        )

    return chunks


def create_or_replace_search_index(
    index_client: SearchIndexClient, index_name: str
) -> None:
    """Delete (if exists) and create an AI Search index with vector support."""
    fields = [
        SimpleField(
            name="id", type=SearchFieldDataType.String, key=True, filterable=True
        ),
        SimpleField(
            name="source", type=SearchFieldDataType.String, filterable=True
        ),
        SearchableField(name="title",   type=SearchFieldDataType.String),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-algo",
                # Tie the profile to the vectorizer so AI Search can vectorize
                # at query-time without a separate embedding call.
                vectorizer_name="openai-vectorizer",
            )
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                vectorizer_name="openai-vectorizer",
                parameters=AzureOpenAIVectorizerParameters(
                    # V2: account endpoint, no Hub path
                    resource_url=FOUNDRY_ACCOUNT_ENDPOINT,
                    deployment_name=EMBEDDING_MODEL,
                    model_name=EMBEDDING_MODEL,
                ),
            )
        ],
    )

    index = SearchIndex(
        name=index_name, fields=fields, vector_search=vector_search
    )

    try:
        index_client.delete_index(index_name)
        print(f"    Deleted existing index '{index_name}'")
    except Exception:
        pass  # Index didn't exist yet – that's fine

    index_client.create_index(index)
    print(f"    Created index '{index_name}'")


def upload_chunks_with_embeddings(chunks: list[dict], openai_client, search_client: SearchClient) -> None:
    """Embed each chunk and batch-upload to AI Search."""
    documents = []
    for chunk in chunks:
        print(f"    Embedding: [{chunk['id']}] '{chunk['title']}'")
        response = openai_client.embeddings.create(
            input=chunk["content"],
            model=EMBEDDING_MODEL,
        )
        documents.append(
            {
                "id":            chunk["id"],
                "title":         chunk["title"],
                "content":       chunk["content"],
                "source":        chunk["source"],
                "contentVector": response.data[0].embedding,
            }
        )

    results = search_client.upload_documents(documents=documents)
    succeeded = sum(1 for r in results if r.succeeded)
    print(f"    Uploaded {len(documents)} docs – {succeeded} succeeded")


# ============================================================================
# Phase 2 helpers: grounded context + responses output parsing
# ============================================================================

def get_grounded_context(search_client: SearchClient, question: str, top_k: int = 3) -> tuple[str, list[tuple[str, str]]]:
    """Fetch top-k search snippets and return (context_blob, source_list)."""
    snippets: list[str] = []
    sources: list[tuple[str, str]] = []

    results = search_client.search(search_text=question, top=top_k)
    for i, doc in enumerate(results, start=1):
        title = str(doc.get("title") or doc.get("source") or f"doc-{i}")
        source = str(doc.get("source") or title)
        content = str(
            doc.get("content")
            or doc.get("chunk")
            or doc.get("text")
            or doc.get("summary")
            or ""
        )
        content = re.sub(r"\s+", " ", content).strip()
        if content:
            snippets.append(f"[{i}] title={title} source={source}\n{content[:1500]}")
            sources.append((title, source))

    if not snippets:
        return "No relevant documents found in Azure AI Search.", []

    return "\n\n".join(snippets), sources


def extract_response_text(response) -> str:
    """Best-effort text extraction from OpenAI responses object."""
    # Handle direct text output
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    
    # Handle structured output with content blocks
    if hasattr(response, "output") and response.output:
        parts = []
        for item in response.output:
            if hasattr(item, "content"):
                for content_block in item.content:
                    if hasattr(content_block, "text"):
                        text = content_block.text
                        if isinstance(text, str):
                            parts.append(text)
                        elif hasattr(text, "value"):
                            parts.append(text.value)
        return "\n".join([p for p in parts if p]).strip()
    
    return ""


def resolve_agent_name(force_new: bool = False) -> str:
    """Resolve a stable agent name, persisting the first generated value locally."""
    if AGENT_NAME:
        return AGENT_NAME

    if not force_new and AGENT_STATE_FILE.exists():
        try:
            state = json.loads(AGENT_STATE_FILE.read_text(encoding="utf-8"))
            existing_name = state.get("agent_name")
            if isinstance(existing_name, str) and existing_name.strip():
                return existing_name.strip()
        except Exception:
            # If state is unreadable, fall back to generating a new stable name.
            pass

    salt = "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
    generated_name = f"RAG-Agent-{salt}"
    AGENT_STATE_FILE.write_text(
        json.dumps({"agent_name": generated_name}, indent=2),
        encoding="utf-8",
    )
    print(f"    Persisted new agent name to {AGENT_STATE_FILE.name}: {generated_name}")
    return generated_name


def resolve_search_connection_id(project_client: AIProjectClient, configured_name: str) -> str:
    """Resolve an AI Search project connection name to the required connection id."""
    try:
        connection = project_client.connections.get(configured_name)
        return connection.id
    except Exception:
        connections = list(project_client.connections.list())
        search_like = [c for c in connections if "search" in (c.name or "").lower()]
        if len(search_like) == 1:
            selected = search_like[0]
            print(
                f"    Configured connection '{configured_name}' not found; "
                f"using detected search connection '{selected.name}'."
            )
            return selected.id

        available = ", ".join(c.name for c in connections) or "(none)"
        raise ValueError(
            "Could not resolve AI Search project connection. "
            f"Set AI_SEARCH_CONNECTION_NAME in .env to one of: {available}"
        )


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 64)
    print("  AI Foundry Agent  ×  AI Search Tool")
    print("  Data: Joakim.md   |  Query: How to reset Robot-Joakim?")
    print("=" * 64)

    # ------------------------------------------------------------------
    # 1. Credential – RBAC only, no account keys
    # ------------------------------------------------------------------
    print("\n[1] Building DefaultAzureCredential (RBAC, no keys)…")
    credential = DefaultAzureCredential(
        additionally_allowed_tenants=[TENANT_ID],
        exclude_interactive_browser_credential=False,
    )

    # ------------------------------------------------------------------
    # Reindex prompt  (--reindex / --no-reindex flags, or interactive)
    # ------------------------------------------------------------------
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--reindex", dest="reindex", action="store_true", default=None)
    _parser.add_argument("--no-reindex", dest="reindex", action="store_false")
    _parser.add_argument("--new-agent", action="store_true", help="Force creating a new agent name and persisting it")
    _args, _ = _parser.parse_known_args()

    if _args.reindex is None:
        # Interactive fallback (used when run directly from a real terminal)
        try:
            reindex = input("\nReindex? (y/n): ").strip().lower()
            do_reindex = reindex == "y"
        except (EOFError, KeyboardInterrupt):
            do_reindex = False
            print("\nReindex: no (non-interactive, defaulting to skip)")
    else:
        do_reindex = _args.reindex
        print(f"\nReindex: {'yes' if do_reindex else 'no'} (CLI flag)")

    # ------------------------------------------------------------------
    # 2. AI Foundry project client  (used for connections + OpenAI client)
    # ------------------------------------------------------------------
    print(f"\n[2] Connecting to AI Foundry project…")
    print(f"    {FOUNDRY_PROJECT_ENDPOINT}")
    project_client = AIProjectClient(
        endpoint=FOUNDRY_PROJECT_ENDPOINT,
        credential=credential,
    )

    # ------------------------------------------------------------------
    # 3. Azure OpenAI client (RBAC) – for embedding generation
    #    AI Foundry V2: models are deployed at the *account* level.
    #    Scope: https://cognitiveservices.azure.com/.default
    #    (the account is a Microsoft.CognitiveServices resource, not AML)
    # ------------------------------------------------------------------
    if do_reindex:
        print("\n[3] Building AzureOpenAI client for embeddings…")
        print(f"    Account endpoint: {FOUNDRY_ACCOUNT_ENDPOINT}")
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        openai_client = AzureOpenAI(
            azure_endpoint=FOUNDRY_ACCOUNT_ENDPOINT,
            azure_ad_token_provider=token_provider,
            api_version="2024-10-21",
        )

        # ------------------------------------------------------------------
        # 4. Parse Joakim.md
        # ------------------------------------------------------------------
        print(f"\n[4] Parsing {DATA_FILE.name}…")
        if not DATA_FILE.exists():
            raise FileNotFoundError(f"Data file missing: {DATA_FILE}")
        chunks = parse_markdown_chunks(DATA_FILE)
        print(f"    {len(chunks)} chunk(s) found:")
        for c in chunks:
            print(f"      [{c['id']}] {c['title']}")

        # ------------------------------------------------------------------
        # 5. Create / replace AI Search index
        # ------------------------------------------------------------------
        print(f"\n[5] Setting up AI Search index '{AI_SEARCH_INDEX_NAME}'…")
        index_client = SearchIndexClient(
            endpoint=AI_SEARCH_ENDPOINT,
            credential=credential,
        )
        create_or_replace_search_index(index_client, AI_SEARCH_INDEX_NAME)

        # ------------------------------------------------------------------
        # 6. Generate embeddings and upload documents
        # ------------------------------------------------------------------
        print(f"\n[6] Indexing chunks with '{EMBEDDING_MODEL}'…")
        search_client = SearchClient(
            endpoint=AI_SEARCH_ENDPOINT,
            index_name=AI_SEARCH_INDEX_NAME,
            credential=credential,
        )
        upload_chunks_with_embeddings(chunks, openai_client, search_client)

        print("    Waiting 15 s for the indexer to settle…")
        time.sleep(15)
    else:
        print("\n[3-6] Skipping reindex – using existing index.")

    # ------------------------------------------------------------------
    # 7. Create AIProjectClient (NEW Foundry pattern)
    # ------------------------------------------------------------------
    print("\n[7] Creating AI Foundry project client…")
    project_client = AIProjectClient(
        endpoint=FOUNDRY_PROJECT_ENDPOINT,
        credential=credential,
    )

    # ------------------------------------------------------------------
    # 8. Set up AI Search tool for the agent
    # ------------------------------------------------------------------
    print("\n[8] Setting up AI Search tool…")
    search_connection_id = resolve_search_connection_id(project_client, AI_SEARCH_CONNECTION_NAME)

    ai_search_tool = AzureAISearchTool(
        azure_ai_search=AzureAISearchToolResource(
            indexes=[
                AISearchIndexResource(
                    project_connection_id=search_connection_id,
                    index_name=AI_SEARCH_INDEX_NAME,
                    query_type=AzureAISearchQueryType.VECTOR,
                    top_k=3,
                )
            ]
        ),
    )

    # ------------------------------------------------------------------
    # 9. Create agent version (NEW pattern: create_version + PromptAgentDefinition)
    # ------------------------------------------------------------------
    print("\n[9] Creating agent version with PromptAgentDefinition…")
    agent_name = resolve_agent_name(force_new=_args.new_agent)
    print(f"    Using agent name: {agent_name}")
    
    agent_definition = PromptAgentDefinition(
        model=CHAT_MODEL,
        instructions=AGENT_INSTRUCTIONS,
        tools=[ai_search_tool],
    )
    
    agent = project_client.agents.create_version(
        agent_name=agent_name,
        definition=agent_definition,
    )
    print(f"    Agent created: {agent_name} (id: {agent.id}, version: {agent.version})")

    # ------------------------------------------------------------------
    # 10. Get OpenAI client and create conversation (NEW pattern)
    # ------------------------------------------------------------------
    print("\n[10] Getting OpenAI client and creating conversation…")
    openai_client = project_client.get_openai_client()
    
    conversation = openai_client.conversations.create()
    print(f"    Conversation created (id: {conversation.id})")

    # ------------------------------------------------------------------
    # 11. Send message to agent via responses.create() with agent_reference
    # ------------------------------------------------------------------
    print(f"\n[11] Sending message: '{FIRST_PROMPT}'")
    
    response = openai_client.responses.create(
        conversation=conversation.id,
        extra_body={
            "agent_reference": {
                "name": agent_name,
                "type": "agent_reference",
            }
        },
        input=FIRST_PROMPT,
    )
    
    answer_text = extract_response_text(response)

    # ------------------------------------------------------------------
    # 12. Print answer
    # ------------------------------------------------------------------
    print("\n[12] Response:")
    print("-" * 64)
    print(answer_text or "(no response text)")
    print("-" * 64)
    
    print("\nDone. Agent created in Foundry (visible in new portal).")


if __name__ == "__main__":
    main()
