# BrainOps Research Agent (Simulated Perplexity Bridge)
# If Perplexity CLI is available, we wrap it. If not, we fall back to Google Search via Gemini.

import os
import sys
import subprocess

def research(query):
    # Check for perplexity cli
    try:
        # Placeholder: Assume 'pplx' is the binary
        # res = subprocess.run(["pplx", query], capture_output=True)
        # return res.stdout
        raise FileNotFoundError # Mocking absence
    except FileNotFoundError:
        # Fallback to Gemini with search capabilities
        print("[System] Perplexity CLI not found. Routing to Gemini Deep Research...")
        cmd = f"gemini -p 'RESEARCH TASK: {query}. Search the web and provide a synthesized answer.'"
        os.system(cmd)

if __name__ == "__main__":
    query = " ".join(sys.argv[1:])
    research(query)
