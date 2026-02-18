#!/usr/bin/env python3
import os
import sys
import argparse
import json
from dotenv import load_dotenv
from supabase import create_client, Client

# Load env from multiple possible locations
load_dotenv('.env')
load_dotenv('.env.local')
load_dotenv('/home/matt-woodworth/dev/_secure/BrainOps.env')

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
# Use Service Role for backend ops, else Anon
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: Supabase credentials not found.")
    sys.exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def store_memory(content: str, agent="cli-orchestrator", tags=None):
    """Stores a memory in the vector store."""
    if tags is None:
        tags = ["cli", "devops"]
    
    # Simple storage for now - in a full implementation, we'd generate embeddings here
    # or rely on a Supabase Edge Function to do it on insert.
    # Assuming 'ai_memories' exists.
    
    payload = {
        "agent_id": agent,
        "key": f"cli-log-{os.urandom(4).hex()}",
        "value": {"type": "cli_context", "text": content},
        "content": content,
        "tags": tags,
        "memory_type": "episodic",
        "is_active": True
    }
    
    data, count = supabase.table("ai_memories").insert(payload).execute()
    print(f"✅ Memory stored: {content[:50]}...")

def recall_memory(query: str, limit=5):
    """Recalls memory. (Mocking vector search with text search for now if embeddings not local)"""
    # In V2, we call the rpc 'match_memories' if it exists, or just text search
    # response = supabase.rpc("match_memories", {"query_embedding": ...}).execute()
    
    # Fallback to simple text search on 'content'
    response = supabase.table("ai_memories")\
        .select("content, created_at")\
        .ilike("content", f"%{query}%")\
        .order("created_at", desc=True)\
        .limit(limit)\
        .execute()
        
    for mem in response.data:
        print(f"[{mem['created_at']}] {mem['content']}")

def list_agents():
    """Lists active agents from activity log."""
    response = supabase.table("agent_activities")\
        .select("agent_id")\
        .order("created_at", desc=True)\
        .limit(50)\
        .execute()
    
    agents = list(set([row['agent_id'] for row in response.data]))
    print(json.dumps(agents, indent=2))

def main():
    parser = argparse.ArgumentParser(description="BrainOps AI Bridge")
    subparsers = parser.add_subparsers(dest="command")

    store_parser = subparsers.add_parser("store", help="Store context")
    store_parser.add_argument("content", type=str)
    
    recall_parser = subparsers.add_parser("recall", help="Recall context")
    recall_parser.add_argument("query", type=str)
    
    subparsers.add_parser("agents", help="List active agents")

    args = parser.parse_args()

    if args.command == "store":
        store_memory(args.content)
    elif args.command == "recall":
        recall_memory(args.query)
    elif args.command == "agents":
        list_agents()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
