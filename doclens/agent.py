"""LangGraph tool-calling agent over the receipts store: Text-to-SQL + semantic item search.

    agent = build_agent(Store())                 # Gemini via langchain-google-genai
    ask(agent, "Which store had the highest average bill?", thread_id="demo")

Per-thread conversation memory comes from LangGraph's checkpointer, so follow-ups such as
"and the lowest?" work within a thread_id.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from doclens.store import SCHEMA_DOC, Store, UnsafeSQL

SYSTEM = f"""You answer questions about a database of scanned restaurant receipts.
{SCHEMA_DOC}

Rules:
- Use run_sql for anything involving counts, sums, averages, dates, stores or amounts. Write SQLite-compatible SQL.
- Use search_items when the user names a dish or product loosely (e.g. "noodles", "iced tea"): it finds
  similar item names, which you can then use in SQL with name IN (...).
- Never invent numbers. If a query returns nothing, say so.
- Answer in one or two sentences and mention amounts in rupiah (Rp)."""


def build_tools(store: Store) -> list:
    @tool
    def run_sql(query: str) -> str:
        """Run one read-only SQL SELECT against the receipts database and return rows as JSON."""
        try:
            rows = store.read_only_query(query)
        except UnsafeSQL as e:
            return f"REJECTED: {e}"
        except Exception as e:  # surface DB errors so the model can fix its SQL
            return f"SQL ERROR: {e}"
        return json.dumps(rows[:50], default=str)

    @tool
    def search_items(text: str, k: int = 5) -> str:
        """Find line items whose names are semantically similar to `text` (embedding search)."""
        return json.dumps(store.search_items(text, k=k), default=str)

    return [run_sql, search_items]


def default_llm():
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"), temperature=0)


def build_agent(store: Store, llm: Any = None):
    tools = build_tools(store)
    model = (llm or default_llm()).bind_tools(tools)

    def agent_node(state: MessagesState) -> Dict[str, List]:
        return {"messages": [model.invoke([SystemMessage(SYSTEM)] + state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=MemorySaver())


def ask(agent, question: str, thread_id: str = "default", recursion_limit: int = 12) -> dict:
    """Returns the answer plus the tool calls made (for transparency / debugging)."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    before = len(agent.get_state(config).values.get("messages", [])) if agent.get_state(config).values else 0
    result = agent.invoke({"messages": [HumanMessage(question)]}, config)
    new = result["messages"][before:]
    calls = [{"tool": c["name"], "args": c["args"]} for m in new if isinstance(m, AIMessage) for c in (m.tool_calls or [])]
    observations = [m.content[:500] for m in new if isinstance(m, ToolMessage)]
    answer = result["messages"][-1].content
    if isinstance(answer, list):  # some providers return content blocks
        answer = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in answer)
    return {"answer": answer, "tool_calls": calls, "observations": observations}
