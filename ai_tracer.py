"""
BrainOps AI Tracer - Deep Observability for Autonomous Agents
Captures thoughts, actions, observations, and state transitions.
"""

import os
import json
import uuid
import logging
import asyncio
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

class SpanType(Enum):
    SESSION = "session"
    THOUGHT = "thought"
    TOOL_CALL = "tool_call"
    OBSERVATION = "observation"
    DECISION = "decision"
    SYSTEM = "system"
    ERROR = "error"

@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    span_type: SpanType
    name: str
    content: str
    metadata: Dict[str, Any]
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"  # running, success, failed

class BrainOpsTracer:
    """
    Distributed tracing specifically designed for AI Agent loops.
    Persists to Postgres for replayability and debugging.
    """

    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
            'database': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
            'password': os.getenv('DB_PASSWORD'),
            'port': int(os.getenv('DB_PORT', 5432))
        }
        self.active_spans: Dict[str, Span] = {}
        self._ensure_schema()

    def _get_connection(self):
        return psycopg2.connect(**self.db_config)

    def _ensure_schema(self):
        """Create tracing tables if they don't exist."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_traces (
                    trace_id UUID PRIMARY KEY,
                    session_id VARCHAR(255),
                    agent_id VARCHAR(255),
                    start_time TIMESTAMPTZ DEFAULT NOW(),
                    end_time TIMESTAMPTZ,
                    status VARCHAR(50),
                    summary TEXT,
                    metadata JSONB DEFAULT '{}'::jsonb
                );
                
                CREATE TABLE IF NOT EXISTS ai_trace_spans (
                    span_id UUID PRIMARY KEY,
                    trace_id UUID REFERENCES ai_traces(trace_id),
                    parent_id UUID,
                    span_type VARCHAR(50),
                    name VARCHAR(255),
                    content TEXT,
                    input_data JSONB,
                    output_data JSONB,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    start_time TIMESTAMPTZ DEFAULT NOW(),
                    end_time TIMESTAMPTZ,
                    duration_ms FLOAT,
                    status VARCHAR(50),
                    error_message TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON ai_trace_spans(trace_id);
                CREATE INDEX IF NOT EXISTS idx_spans_type ON ai_trace_spans(span_type);
            """)
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to init tracer schema: {e}")

    def start_trace(self, session_id: str, agent_id: str, metadata: Dict = None) -> str:
        """Start a new high-level trace (e.g., a user request or cron job)."""
        trace_id = str(uuid.uuid4())
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ai_traces (trace_id, session_id, agent_id, status, metadata)
                VALUES (%s, %s, %s, 'running', %s)
            """, (trace_id, session_id, agent_id, json.dumps(metadata or {})))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to start trace: {e}")
            
        return trace_id

    def end_trace(self, trace_id: str, status: str = "completed", summary: str = None):
        """Complete a trace."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                UPDATE ai_traces 
                SET status = %s, end_time = NOW(), summary = %s
                WHERE trace_id = %s
            """, (status, summary, trace_id))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to end trace: {e}")

    def start_span(self, trace_id: str, name: str, span_type: SpanType, 
                   parent_id: Optional[str] = None, content: str = "", 
                   inputs: Dict = None) -> str:
        """Start a granular unit of work (thought, tool call, etc)."""
        span_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            span_type=span_type,
            name=name,
            content=content,
            metadata=inputs or {},
            start_time=start_time.timestamp()
        )
        self.active_spans[span_id] = span
        
        # Async persist to DB to not block execution? 
        # For reliability in this environment, we'll do sync blocking insert for now.
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ai_trace_spans (
                    span_id, trace_id, parent_id, span_type, name, content, 
                    input_data, start_time, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'running')
            """, (
                span_id, trace_id, parent_id, span_type.value, name, content,
                json.dumps(inputs or {}), start_time
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to start span: {e}")
            
        return span_id

    def end_span(self, span_id: str, output: Any = None, status: str = "success", 
                 error: str = None):
        """Complete a span with results."""
        if span_id not in self.active_spans:
            logger.warning(f"Span {span_id} not found in memory")
            return

        span = self.active_spans.pop(span_id)
        end_time = datetime.now()
        duration_ms = (end_time.timestamp() - span.start_time) * 1000
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                UPDATE ai_trace_spans 
                SET end_time = %s, duration_ms = %s, status = %s, 
                    output_data = %s, error_message = %s
                WHERE span_id = %s
            """, (
                end_time, duration_ms, status, 
                json.dumps(output) if output else None, 
                error, span_id
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to end span: {e}")

    # Context Manager Support
    def span(self, trace_id: str, name: str, span_type: SpanType, parent_id: str = None, **kwargs):
        return TraceContext(self, trace_id, name, span_type, parent_id, **kwargs)

class TraceContext:
    def __init__(self, tracer: BrainOpsTracer, trace_id: str, name: str, 
                 span_type: SpanType, parent_id: str = None, **kwargs):
        self.tracer = tracer
        self.trace_id = trace_id
        self.name = name
        self.span_type = span_type
        self.parent_id = parent_id
        self.kwargs = kwargs
        self.span_id = None

    def __enter__(self):
        self.span_id = self.tracer.start_span(
            self.trace_id, self.name, self.span_type, self.parent_id, inputs=self.kwargs
        )
        return self.span_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "failed" if exc_type else "success"
        error = str(exc_val) if exc_val else None
        self.tracer.end_span(self.span_id, status=status, error=error)
