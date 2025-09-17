#!/usr/bin/env python3
"""
Conversation Memory Persistence System
Stores and retrieves all conversation context across sessions
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor
import openai
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD", "REDACTED_SUPABASE_DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", 5432))
}

# OpenAI configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

class MessageRole(Enum):
    """Roles in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    FUNCTION = "function"

class ConversationStatus(Enum):
    """Status of conversation"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

@dataclass
class Message:
    """Represents a single message in conversation"""
    id: str
    conversation_id: str
    role: MessageRole
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    token_count: int
    embedding: Optional[List[float]] = None

@dataclass
class Conversation:
    """Represents a full conversation"""
    id: str
    user_id: str
    title: str
    status: ConversationStatus
    started_at: datetime
    last_message_at: datetime
    message_count: int
    total_tokens: int
    context: Dict[str, Any]
    summary: Optional[str] = None
    topics: List[str] = None
    sentiment: float = 0.0

class ConversationMemory:
    """Manages conversation persistence and retrieval"""

    def __init__(self):
        """Initialize the conversation memory system"""
        self._ensure_tables()
        self.active_conversations = {}
        self.context_window = 10  # Number of recent messages to maintain in context

    def _ensure_tables(self):
        """Create necessary database tables"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(255) NOT NULL,
                    title VARCHAR(500),
                    status VARCHAR(20) DEFAULT 'active',
                    started_at TIMESTAMPTZ DEFAULT NOW(),
                    last_message_at TIMESTAMPTZ DEFAULT NOW(),
                    message_count INT DEFAULT 0,
                    total_tokens INT DEFAULT 0,
                    context JSONB DEFAULT '{}'::jsonb,
                    summary TEXT,
                    topics JSONB DEFAULT '[]'::jsonb,
                    sentiment FLOAT DEFAULT 0.0,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    token_count INT DEFAULT 0,
                    embedding vector(1536),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Conversation snapshots for long-term memory
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_snapshots (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
                    snapshot_data JSONB NOT NULL,
                    message_range_start INT,
                    message_range_end INT,
                    summary TEXT,
                    key_points JSONB DEFAULT '[]'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Conversation links for cross-referencing
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_links (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source_conversation_id UUID REFERENCES conversations(id),
                    target_conversation_id UUID REFERENCES conversations(id),
                    link_type VARCHAR(50),
                    strength FLOAT DEFAULT 0.5,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_user_id
                ON conversations(user_id);

                CREATE INDEX IF NOT EXISTS idx_conv_status
                ON conversations(status);

                CREATE INDEX IF NOT EXISTS idx_conv_last_message
                ON conversations(last_message_at DESC);

                CREATE INDEX IF NOT EXISTS idx_msg_conversation
                ON conversation_messages(conversation_id);

                CREATE INDEX IF NOT EXISTS idx_msg_timestamp
                ON conversation_messages(timestamp DESC);
            """)

            conn.commit()
            cursor.close()
            conn.close()

            logger.info("Conversation memory tables initialized")

        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def start_conversation(self, user_id: str, title: str = None, context: Dict[str, Any] = None) -> str:
        """Start a new conversation"""
        try:
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            # Auto-generate title if not provided
            if not title:
                title = f"Conversation - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            cursor.execute("""
                INSERT INTO conversations (user_id, title, context)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (user_id, title, json.dumps(context or {})))

            result = cursor.fetchone()
            conversation_id = str(result['id'])

            conn.commit()
            cursor.close()
            conn.close()

            # Cache in memory
            self.active_conversations[conversation_id] = {
                'user_id': user_id,
                'messages': [],
                'context': context or {}
            }

            logger.info(f"Started conversation: {conversation_id}")
            return conversation_id

        except Exception as e:
            logger.error(f"Failed to start conversation: {e}")
            return None

    def add_message(self,
                   conversation_id: str,
                   role: MessageRole,
                   content: str,
                   metadata: Dict[str, Any] = None) -> Optional[str]:
        """Add a message to conversation"""
        try:
            # Calculate token count (approximate)
            token_count = len(content.split()) * 1.3

            # Generate embedding for important messages
            embedding = None
            if role in [MessageRole.USER, MessageRole.ASSISTANT]:
                embedding = self._generate_embedding(content)

            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            # Insert message
            embedding_str = None
            if embedding:
                embedding_str = '[' + ','.join(map(str, embedding)) + ']'

            cursor.execute("""
                INSERT INTO conversation_messages
                (conversation_id, role, content, metadata, token_count, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                RETURNING id
            """, (
                conversation_id,
                role.value,
                content,
                json.dumps(metadata or {}),
                int(token_count),
                embedding_str
            ))

            message_id = str(cursor.fetchone()['id'])

            # Update conversation stats
            cursor.execute("""
                UPDATE conversations
                SET message_count = message_count + 1,
                    total_tokens = total_tokens + %s,
                    last_message_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s
            """, (int(token_count), conversation_id))

            # Check if we need to create a snapshot
            cursor.execute("""
                SELECT message_count FROM conversations WHERE id = %s
            """, (conversation_id,))

            message_count = cursor.fetchone()['message_count']

            # Create snapshot every 50 messages
            if message_count % 50 == 0:
                self._create_snapshot(cursor, conversation_id, message_count)

            conn.commit()
            cursor.close()
            conn.close()

            # Update memory cache
            if conversation_id in self.active_conversations:
                self.active_conversations[conversation_id]['messages'].append({
                    'role': role.value,
                    'content': content,
                    'timestamp': datetime.now(timezone.utc)
                })

            return message_id

        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            return None

    def get_conversation_context(self,
                                 conversation_id: str,
                                 num_messages: int = None) -> Dict[str, Any]:
        """Get conversation context with recent messages"""
        try:
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            # Get conversation details
            cursor.execute("""
                SELECT * FROM conversations WHERE id = %s
            """, (conversation_id,))

            conversation = cursor.fetchone()
            if not conversation:
                return None

            # Get recent messages
            num_messages = num_messages or self.context_window
            cursor.execute("""
                SELECT role, content, metadata, timestamp, token_count
                FROM conversation_messages
                WHERE conversation_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (conversation_id, num_messages))

            messages = cursor.fetchall()
            messages.reverse()  # Put in chronological order

            # Get any relevant snapshots
            cursor.execute("""
                SELECT summary, key_points
                FROM conversation_snapshots
                WHERE conversation_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (conversation_id,))

            snapshot = cursor.fetchone()

            cursor.close()
            conn.close()

            return {
                'conversation_id': conversation_id,
                'user_id': conversation['user_id'],
                'title': conversation['title'],
                'status': conversation['status'],
                'message_count': conversation['message_count'],
                'total_tokens': conversation['total_tokens'],
                'messages': messages,
                'context': conversation['context'],
                'summary': conversation['summary'],
                'topics': conversation['topics'],
                'snapshot': snapshot
            }

        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return None

    def search_conversations(self,
                           user_id: str,
                           query: str,
                           limit: int = 10) -> List[Dict]:
        """Search through conversation history"""
        try:
            # Generate query embedding
            embedding = self._generate_embedding(query)
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'

            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            # Search messages by similarity
            cursor.execute("""
                WITH relevant_messages AS (
                    SELECT
                        conversation_id,
                        content,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM conversation_messages
                    WHERE embedding IS NOT NULL
                        AND conversation_id IN (
                            SELECT id FROM conversations WHERE user_id = %s
                        )
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                )
                SELECT DISTINCT
                    c.id,
                    c.title,
                    c.summary,
                    c.last_message_at,
                    c.message_count,
                    MAX(rm.similarity) as relevance
                FROM conversations c
                JOIN relevant_messages rm ON c.id = rm.conversation_id
                GROUP BY c.id, c.title, c.summary, c.last_message_at, c.message_count
                ORDER BY relevance DESC
            """, (embedding_str, user_id, embedding_str, limit * 3))

            results = cursor.fetchall()

            cursor.close()
            conn.close()

            return results[:limit]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_related_conversations(self, conversation_id: str) -> List[Dict]:
        """Get conversations related to the current one"""
        try:
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            # Get linked conversations
            cursor.execute("""
                SELECT
                    c.id,
                    c.title,
                    c.summary,
                    cl.link_type,
                    cl.strength
                FROM conversation_links cl
                JOIN conversations c ON (
                    cl.target_conversation_id = c.id
                    OR cl.source_conversation_id = c.id
                )
                WHERE (cl.source_conversation_id = %s
                    OR cl.target_conversation_id = %s)
                    AND c.id != %s
                ORDER BY cl.strength DESC
                LIMIT 5
            """, (conversation_id, conversation_id, conversation_id))

            related = cursor.fetchall()

            cursor.close()
            conn.close()

            return related

        except Exception as e:
            logger.error(f"Failed to get related conversations: {e}")
            return []

    def _create_snapshot(self, cursor, conversation_id: str, message_count: int):
        """Create a snapshot of conversation for long-term storage"""
        try:
            # Get messages for snapshot
            start_range = max(0, message_count - 50)

            cursor.execute("""
                SELECT role, content, timestamp
                FROM conversation_messages
                WHERE conversation_id = %s
                ORDER BY timestamp
                OFFSET %s LIMIT 50
            """, (conversation_id, start_range))

            messages = cursor.fetchall()

            # Generate summary using AI
            content_for_summary = "\n".join([
                f"{msg['role']}: {msg['content'][:200]}"
                for msg in messages[:10]
            ])

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Summarize this conversation segment and extract key points."},
                    {"role": "user", "content": content_for_summary}
                ],
                temperature=0.3,
                max_tokens=200
            )

            summary = response.choices[0].message.content

            # Extract key points
            key_points = self._extract_key_points(messages)

            # Store snapshot
            cursor.execute("""
                INSERT INTO conversation_snapshots
                (conversation_id, snapshot_data, message_range_start,
                 message_range_end, summary, key_points)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                conversation_id,
                json.dumps([msg for msg in messages]),
                start_range,
                message_count,
                summary,
                json.dumps(key_points)
            ))

            logger.info(f"Created snapshot for conversation {conversation_id}")

        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")

    def _extract_key_points(self, messages: List[Dict]) -> List[str]:
        """Extract key points from messages"""
        key_points = []

        for msg in messages:
            content = msg['content'].lower()
            # Look for important indicators
            if any(indicator in content for indicator in ['important', 'remember', 'note', 'key', 'critical']):
                # Extract sentence containing the indicator
                sentences = msg['content'].split('.')
                for sentence in sentences:
                    if any(indicator in sentence.lower() for indicator in ['important', 'remember', 'note']):
                        key_points.append(sentence.strip())
                        break

        return key_points[:5]  # Limit to 5 key points

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text[:8000]  # Limit to avoid token limits
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * 1536

    def end_conversation(self, conversation_id: str) -> bool:
        """Mark conversation as completed"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Generate final summary
            cursor.execute("""
                SELECT role, content
                FROM conversation_messages
                WHERE conversation_id = %s
                ORDER BY timestamp
            """, (conversation_id,))

            messages = cursor.fetchall()

            if messages:
                # Use AI to generate summary
                content = "\n".join([f"{msg[0]}: {msg[1][:100]}" for msg in messages[:20]])

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Provide a concise summary of this conversation."},
                        {"role": "user", "content": content}
                    ],
                    temperature=0.3,
                    max_tokens=150
                )

                summary = response.choices[0].message.content

                # Update conversation
                cursor.execute("""
                    UPDATE conversations
                    SET status = 'completed',
                        summary = %s,
                        updated_at = NOW()
                    WHERE id = %s
                """, (summary, conversation_id))

            conn.commit()
            cursor.close()
            conn.close()

            # Remove from active cache
            if conversation_id in self.active_conversations:
                del self.active_conversations[conversation_id]

            return True

        except Exception as e:
            logger.error(f"Failed to end conversation: {e}")
            return False

    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get conversation statistics for a user"""
        try:
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_conversations,
                    SUM(message_count) as total_messages,
                    SUM(total_tokens) as total_tokens,
                    AVG(message_count) as avg_messages_per_conversation,
                    MAX(last_message_at) as last_active
                FROM conversations
                WHERE user_id = %s
            """, (user_id,))

            stats = cursor.fetchone()

            # Get topic distribution
            cursor.execute("""
                SELECT topics
                FROM conversations
                WHERE user_id = %s AND topics IS NOT NULL
            """, (user_id,))

            all_topics = []
            for row in cursor.fetchall():
                if row['topics']:
                    all_topics.extend(row['topics'])

            # Count topic frequency
            topic_counts = {}
            for topic in all_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1

            stats['top_topics'] = sorted(
                topic_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            cursor.close()
            conn.close()

            return stats

        except Exception as e:
            logger.error(f"Failed to get user statistics: {e}")
            return {}

# Global instance - create lazily
conversation_memory = None

def get_conversation_memory():
    """Get or create conversation memory instance"""
    global conversation_memory
    if conversation_memory is None:
        conversation_memory = ConversationMemory()
    return conversation_memory

# Example usage
if __name__ == "__main__":
    cm = get_conversation_memory()

    # Start a conversation
    user_id = "test_user_123"
    conv_id = cm.start_conversation(user_id, "Test Conversation")

    # Add messages
    cm.add_message(conv_id, MessageRole.USER, "Hello, how are you?")
    cm.add_message(conv_id, MessageRole.ASSISTANT, "I'm doing well, thank you! How can I help you today?")
    cm.add_message(conv_id, MessageRole.USER, "Can you explain how conversation memory works?")

    # Get context
    context = cm.get_conversation_context(conv_id)
    print(f"Conversation context: {context}")

    # Search conversations
    results = cm.search_conversations(user_id, "memory")
    print(f"Search results: {results}")

    # End conversation
    cm.end_conversation(conv_id)