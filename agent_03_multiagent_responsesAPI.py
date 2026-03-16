"""Multi-agent orchestrator using persisted Foundry agents (NEW pattern).

Pipeline:
    Router -> (Coding | RAG) -> Presenter

Each role is a persisted Foundry agent version created via create_version().
Each invocation uses responses.create(...) with agent_reference.
"""

import argparse
import importlib.util
import json
import os
import random
import re
import string
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
        AISearchIndexResource,
        AzureAISearchQueryType,
        AzureAISearchTool,
        AzureAISearchToolResource,
        CodeInterpreterTool,
        PromptAgentDefinition,
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv


# ============================================================================
# Configuration
# ============================================================================

_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(_BASE_DIR.parent / ".env")


def _env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key, default)
    return None if value == "<TODO>" else value


SUBSCRIPTION_ID = _env("SUBSCRIPTION_ID", "<TODO>")
TENANT_ID = _env("TENANT_ID", "<TODO>")
RESOURCE_GROUP = _env("RESOURCE_GROUP", "<TODO>")

FOUNDRY_PROJECT_ENDPOINT = _env("FOUNDRY_PROJECT_ENDPOINT", "<TODO>")

AI_SEARCH_NAME = _env("AI_SEARCH_NAME", "<TODO>")
AI_SEARCH_ENDPOINT = f"https://{AI_SEARCH_NAME}.search.windows.net" if AI_SEARCH_NAME else None
AI_SEARCH_INDEX_NAME = _env("AI_SEARCH_INDEX_NAME", "<TODO>")

CHAT_MODEL = _env("CHAT_MODEL", "<TODO>")
DEFAULT_QUERY = _env("DEFAULT_QUERY", "<TODO>") or "How to reset Robot-Joakim?"
AI_SEARCH_CONNECTION_NAME = _env("AI_SEARCH_CONNECTION_NAME", "Connection-AzureAISearch-Foundry")

ROUTER_AGENT_NAME = _env("ROUTER_AGENT_NAME", None)
CODING_AGENT_NAME = _env("CODING_AGENT_NAME", None)
RAG_AGENT_NAME = _env("RAG_AGENT_NAME", None)
PRESENTER_AGENT_NAME = _env("PRESENTER_AGENT_NAME", None)

AGENT_STATE_FILE = Path(__file__).with_name(".agent_03_multiagent_state.json")


# ============================================================================
# Load prompt modules
# ============================================================================

_MA_DIR = Path(__file__).parent / "multi-agent"


def _load_module(filename: str):
    path = _MA_DIR / filename
    mod_name = filename.replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


router_mod = _load_module("01_router_agent_salt.py")
coding_mod = _load_module("02a_coding_agent_salt.py")
rag_mod = _load_module("02b_RAG_agent_salt.py")
presenter_mod = _load_module("03_presenter_agent.py")


# ============================================================================
# Helpers
# ============================================================================

def _make_salt(k: int = 5) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=k))


def _extract_output_text(resp) -> str:
    """Best-effort extraction for OpenAI/Azure Responses API result text."""
    text = getattr(resp, "output_text", None)
    if text:
        return text

    parts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if isinstance(value, str):
                parts.append(value)
            elif hasattr(value, "value"):
                parts.append(value.value)
    return "\n".join([p for p in parts if p]).strip()


def _run_single_turn(
    openai_client,
    agent_name: str,
    user_message: str,
) -> str:
    conversation = openai_client.conversations.create()
    resp = openai_client.responses.create(
        conversation=conversation.id,
        extra_body={
            "agent_reference": {
                "name": agent_name,
                "type": "agent_reference",
            }
        },
        input=user_message,
    )
    return _extract_output_text(resp)


def _parse_json_route(router_text: str) -> tuple[str, str]:
    try:
        data = json.loads(router_text.strip())
        return data.get("route", "RAG_AGENT"), data.get("reason", "")
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^{}]*"route"[^{}]*\}', router_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return data.get("route", "RAG_AGENT"), data.get("reason", "")
        except json.JSONDecodeError:
            pass

    if "CODING" in router_text.upper():
        return "CODING_AGENT", "(heuristic)"
    return "RAG_AGENT", "(heuristic fallback)"


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


def resolve_agent_names(force_new: bool = False) -> dict[str, str]:
    """Resolve stable agent names from env/state or generate and persist new names."""
    prefixes = {
        "router": router_mod.ROUTER_AGENT_NAME_PREFIX,
        "coding": coding_mod.CODING_AGENT_NAME_PREFIX,
        "rag": rag_mod.RAG_AGENT_NAME_PREFIX,
        "presenter": presenter_mod.PRESENTER_AGENT_NAME_PREFIX,
    }
    configured = {
        "router": ROUTER_AGENT_NAME,
        "coding": CODING_AGENT_NAME,
        "rag": RAG_AGENT_NAME,
        "presenter": PRESENTER_AGENT_NAME,
    }

    persisted: dict[str, str] = {}
    if not force_new and AGENT_STATE_FILE.exists():
        try:
            candidate = json.loads(AGENT_STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                persisted = {k: v for k, v in candidate.items() if isinstance(v, str) and v.strip()}
        except Exception:
            persisted = {}

    names: dict[str, str] = {}
    generated_any = False
    salt = _make_salt(5)

    for key, prefix in prefixes.items():
        if configured.get(key):
            names[key] = configured[key]
            continue
        if not force_new and persisted.get(key):
            names[key] = persisted[key]
            continue
        names[key] = f"{prefix}-{salt}"
        generated_any = True

    if generated_any or (not AGENT_STATE_FILE.exists() and not any(configured.values())):
        AGENT_STATE_FILE.write_text(json.dumps(names, indent=2), encoding="utf-8")
        print(f"    Persisted multi-agent names to {AGENT_STATE_FILE.name}")

    return names


def create_agent_versions(
    project_client: AIProjectClient,
    names: dict[str, str],
    search_connection_id: str,
) -> dict[str, object]:
    """Create a new version for each orchestrator role under stable agent names."""
    router_definition = PromptAgentDefinition(
        model=CHAT_MODEL,
        instructions=router_mod.ROUTER_AGENT_INSTRUCTIONS,
    )

    coding_definition = PromptAgentDefinition(
        model=CHAT_MODEL,
        instructions=coding_mod.CODING_AGENT_INSTRUCTIONS,
        tools=[CodeInterpreterTool()],
    )

    rag_tool = AzureAISearchTool(
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
    rag_definition = PromptAgentDefinition(
        model=CHAT_MODEL,
        instructions=rag_mod.RAG_AGENT_INSTRUCTIONS,
        tools=[rag_tool],
    )

    presenter_definition = PromptAgentDefinition(
        model=CHAT_MODEL,
        instructions=presenter_mod.PRESENTER_AGENT_INSTRUCTIONS,
        tools=[CodeInterpreterTool()],
    )

    return {
        "router": project_client.agents.create_version(agent_name=names["router"], definition=router_definition),
        "coding": project_client.agents.create_version(agent_name=names["coding"], definition=coding_definition),
        "rag": project_client.agents.create_version(agent_name=names["rag"], definition=rag_definition),
        "presenter": project_client.agents.create_version(
            agent_name=names["presenter"], definition=presenter_definition
        ),
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Responses API Multi-Agent Orchestrator")
    parser.add_argument(
        "--query",
        default=None,
        help=f'User query (default: ask interactively; or "{DEFAULT_QUERY}")',
    )
    parser.add_argument(
        "--new-agent",
        action="store_true",
        help="Force new multi-agent names (and persist them) when env names are not set.",
    )
    args = parser.parse_args()

    print("=" * 68)
    print("  Multi-Agent Orchestrator (Responses API)")
    print("  Router -> (CodingAgent | RAGAgent) -> PresenterAgent")
    print("=" * 68)

    if args.query:
        user_query = args.query
    else:
        try:
            user_query = input("\nEnter your query (or press Enter for default):\n> ").strip()
            if not user_query:
                user_query = DEFAULT_QUERY
        except (EOFError, KeyboardInterrupt):
            user_query = DEFAULT_QUERY
    print(f'\n  Query: "{user_query}"')

    print("\n[1] DefaultAzureCredential (RBAC)...")
    credential = DefaultAzureCredential(
        additionally_allowed_tenants=[TENANT_ID] if TENANT_ID else None,
        exclude_interactive_browser_credential=False,
    )

    print("\n[2] Connecting to AI Foundry project...")
    print(f"    {FOUNDRY_PROJECT_ENDPOINT}")
    project_client = AIProjectClient(
        endpoint=FOUNDRY_PROJECT_ENDPOINT,
        credential=credential,
    )

    print("\n[3] Resolving AI Search connection id...")
    search_connection_id = resolve_search_connection_id(project_client, AI_SEARCH_CONNECTION_NAME)

    print("\n[4] Resolving stable multi-agent names...")
    names = resolve_agent_names(force_new=args.new_agent)
    print(f"    [01] router    -> {names['router']}")
    print(f"    [02a] coding   -> {names['coding']}")
    print(f"    [02b] rag      -> {names['rag']}")
    print(f"    [03] presenter -> {names['presenter']}")

    print("\n[5] Creating agent versions (create_version)...")
    created = create_agent_versions(project_client, names, search_connection_id)
    print(f"    Router    : id={created['router'].id} version={created['router'].version}")
    print(f"    Coding    : id={created['coding'].id} version={created['coding'].version}")
    print(f"    RAG       : id={created['rag'].id} version={created['rag'].version}")
    print(f"    Presenter : id={created['presenter'].id} version={created['presenter'].version}")

    openai_client = project_client.get_openai_client()

    print("\n[6] Router Agent -> routing query...")
    router_text = _run_single_turn(
        openai_client=openai_client,
        agent_name=names["router"],
        user_message=user_query,
    )
    print(f"    Raw router response: {router_text[:200]}")

    route, reason = _parse_json_route(router_text)
    print(f"    Route: {route}  ({reason})")

    specialist_text = ""
    if route == "CODING_AGENT":
        print("\n[7] Coding Agent -> running specialist response...")
        specialist_text = _run_single_turn(
            openai_client=openai_client,
            agent_name=names["coding"],
            user_message=user_query,
        )
    else:
        print("\n[7] RAG Agent -> running specialist response with Azure AI Search tool...")
        specialist_text = _run_single_turn(
            openai_client=openai_client,
            agent_name=names["rag"],
            user_message=user_query,
        )

    print(f"    Specialist response ({len(specialist_text)} chars)")

    print(f"\n[8] Presenter Agent -> formatting output (SOURCE={route})...")
    presenter_prompt = (
        f"SOURCE={route}\n\n"
        f"ORIGINAL USER QUESTION:\n{user_query}\n\n"
        f"SPECIALIST AGENT RESPONSE:\n{specialist_text}"
    )
    presenter_text = _run_single_turn(
        openai_client=openai_client,
        agent_name=names["presenter"],
        user_message=presenter_prompt,
    )

    print("\n" + "=" * 68)
    print(f"  FINAL OUTPUT  (formatted by Presenter Agent | route={route})")
    print("=" * 68)

    if route == "RAG_AGENT":
        raw = presenter_text.strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                print(json.dumps(parsed, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(raw)
        else:
            print(raw)
    else:
        print(presenter_text)

    print("=" * 68)
    print("\nDone")


if __name__ == "__main__":
    main()
