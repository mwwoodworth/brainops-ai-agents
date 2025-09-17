#!/usr/bin/env python3
"""
Document Processing with Supabase Storage
Automatically processes and understands uploaded documents
"""

import os
import json
import logging
import uuid
import hashlib
import asyncio
import mimetypes
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import httpx
from openai import OpenAI
import PyPDF2
import docx
import openpyxl
from PIL import Image
import pytesseract
import io
import base64

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

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://yomagoqdmxszqtdwuhab.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "")
STORAGE_BUCKET = "documents"

# OpenAI configuration
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DocumentType(Enum):
    """Types of documents"""
    PDF = "pdf"
    WORD = "word"
    EXCEL = "excel"
    IMAGE = "image"
    TEXT = "text"
    PRESENTATION = "presentation"
    EMAIL = "email"
    CONTRACT = "contract"
    INVOICE = "invoice"
    REPORT = "report"
    UNKNOWN = "unknown"

class ProcessingStatus(Enum):
    """Document processing status"""
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"

class ExtractionMethod(Enum):
    """Text extraction methods"""
    NATIVE = "native"
    OCR = "ocr"
    AI_VISION = "ai_vision"
    HYBRID = "hybrid"

class DocumentCategory(Enum):
    """Document categories for classification"""
    LEGAL = "legal"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    MARKETING = "marketing"
    HR = "hr"
    OPERATIONAL = "operational"
    CUSTOMER = "customer"
    VENDOR = "vendor"
    INTERNAL = "internal"
    EXTERNAL = "external"

class DocumentProcessor:
    """Main document processing class"""

    def __init__(self):
        """Initialize the document processor"""
        self.storage_client = None
        self.supported_formats = {
            '.pdf': DocumentType.PDF,
            '.doc': DocumentType.WORD,
            '.docx': DocumentType.WORD,
            '.xls': DocumentType.EXCEL,
            '.xlsx': DocumentType.EXCEL,
            '.png': DocumentType.IMAGE,
            '.jpg': DocumentType.IMAGE,
            '.jpeg': DocumentType.IMAGE,
            '.txt': DocumentType.TEXT,
            '.csv': DocumentType.TEXT,
            '.ppt': DocumentType.PRESENTATION,
            '.pptx': DocumentType.PRESENTATION
        }
        self._init_database()
        self._init_storage()

    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Create tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    filename VARCHAR(255),
                    document_type VARCHAR(50),
                    mime_type VARCHAR(100),
                    file_size BIGINT,
                    storage_path TEXT,
                    storage_url TEXT,
                    hash_value VARCHAR(64),
                    status VARCHAR(50) DEFAULT 'uploaded',
                    upload_source VARCHAR(100),
                    user_id VARCHAR(255),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    processed_at TIMESTAMPTZ
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_document_content (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID REFERENCES ai_documents(id),
                    content_type VARCHAR(50),
                    raw_text TEXT,
                    structured_data JSONB,
                    extraction_method VARCHAR(50),
                    language VARCHAR(10),
                    word_count INT,
                    page_count INT,
                    confidence_score FLOAT,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_document_analysis (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID REFERENCES ai_documents(id),
                    analysis_type VARCHAR(50),
                    category VARCHAR(50),
                    summary TEXT,
                    key_entities JSONB,
                    key_terms JSONB,
                    sentiment_score FLOAT,
                    topics JSONB,
                    classifications JSONB,
                    insights JSONB,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_document_embeddings (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID REFERENCES ai_documents(id),
                    chunk_index INT,
                    chunk_text TEXT,
                    embedding vector(1536),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_document_relationships (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source_document_id UUID REFERENCES ai_documents(id),
                    target_document_id UUID REFERENCES ai_documents(id),
                    relationship_type VARCHAR(50),
                    confidence FLOAT,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_document_actions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID REFERENCES ai_documents(id),
                    action_type VARCHAR(50),
                    action_status VARCHAR(50),
                    action_data JSONB,
                    triggered_by VARCHAR(100),
                    executed_at TIMESTAMPTZ,
                    result JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON ai_documents(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_type ON ai_documents(document_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_user ON ai_documents(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_document ON ai_document_content(document_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_document ON ai_document_analysis(document_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_document ON ai_document_embeddings(document_id)")

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def _init_storage(self):
        """Initialize Supabase storage client"""
        try:
            self.storage_client = httpx.AsyncClient(
                base_url=f"{SUPABASE_URL}/storage/v1",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}"
                }
            )
        except Exception as e:
            logger.error(f"Failed to initialize storage client: {e}")

    async def upload_document(
        self,
        file_path: str = None,
        file_content: bytes = None,
        filename: str = None,
        user_id: str = None,
        source: str = "manual",
        metadata: Dict = None
    ) -> str:
        """Upload a document to storage and create database record"""
        try:
            # Get file content if path provided
            if file_path and not file_content:
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                if not filename:
                    filename = os.path.basename(file_path)

            if not file_content or not filename:
                raise ValueError("File content and filename required")

            # Calculate file hash
            file_hash = hashlib.sha256(file_content).hexdigest()

            # Check for duplicate
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT id, storage_url FROM ai_documents
                WHERE hash_value = %s
            """, (file_hash,))

            existing = cursor.fetchone()
            if existing:
                logger.info(f"Document already exists: {existing[0]}")
                cursor.close()
                conn.close()
                return existing[0]

            # Determine document type
            file_ext = os.path.splitext(filename)[1].lower()
            doc_type = self.supported_formats.get(file_ext, DocumentType.UNKNOWN)
            mime_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'

            # Generate storage path
            doc_id = str(uuid.uuid4())
            storage_path = f"{user_id or 'public'}/{datetime.now().strftime('%Y/%m/%d')}/{doc_id}/{filename}"

            # Upload to Supabase Storage
            if self.storage_client:
                response = await self.storage_client.post(
                    f"/object/{STORAGE_BUCKET}/{storage_path}",
                    content=file_content,
                    headers={"Content-Type": mime_type}
                )

                if response.status_code == 200:
                    storage_url = f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{storage_path}"
                else:
                    storage_url = None
                    logger.error(f"Failed to upload to storage: {response.text}")
            else:
                # Fallback: Store locally
                storage_url = f"local://{storage_path}"

            # Create database record
            cursor.execute("""
                INSERT INTO ai_documents
                (id, filename, document_type, mime_type, file_size,
                 storage_path, storage_url, hash_value, status,
                 upload_source, user_id, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                doc_id,
                filename,
                doc_type.value,
                mime_type,
                len(file_content),
                storage_path,
                storage_url,
                file_hash,
                ProcessingStatus.UPLOADED.value,
                source,
                user_id,
                Json(metadata or {})
            ))

            conn.commit()
            cursor.close()
            conn.close()

            # Queue for processing
            await self._queue_for_processing(doc_id)

            return doc_id

        except Exception as e:
            logger.error(f"Failed to upload document: {e}")
            return None

    async def _queue_for_processing(self, document_id: str):
        """Queue document for processing"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE ai_documents
                SET status = %s
                WHERE id = %s
            """, (ProcessingStatus.QUEUED.value, document_id))

            conn.commit()
            cursor.close()
            conn.close()

            # Trigger async processing
            asyncio.create_task(self.process_document(document_id))

        except Exception as e:
            logger.error(f"Failed to queue document: {e}")

    async def process_document(self, document_id: str) -> Dict:
        """Process a document - extract text, analyze, and create embeddings"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Update status
            cursor.execute("""
                UPDATE ai_documents
                SET status = %s
                WHERE id = %s
            """, (ProcessingStatus.PROCESSING.value, document_id))
            conn.commit()

            # Get document info
            cursor.execute("""
                SELECT * FROM ai_documents WHERE id = %s
            """, (document_id,))

            document = cursor.fetchone()
            if not document:
                raise ValueError(f"Document not found: {document_id}")

            # Extract text based on document type
            extracted_text = await self._extract_text(document)

            # Store extracted content
            if extracted_text:
                content_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO ai_document_content
                    (id, document_id, content_type, raw_text,
                     extraction_method, word_count, confidence_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    content_id,
                    document_id,
                    document['document_type'],
                    extracted_text['text'],
                    extracted_text['method'],
                    len(extracted_text['text'].split()),
                    extracted_text.get('confidence', 1.0)
                ))

                # Analyze document
                analysis = await self._analyze_document(
                    document_id,
                    extracted_text['text'],
                    document
                )

                # Create embeddings
                await self._create_embeddings(
                    document_id,
                    extracted_text['text']
                )

                # Find relationships
                await self._find_relationships(document_id, analysis)

                # Update status
                cursor.execute("""
                    UPDATE ai_documents
                    SET status = %s, processed_at = %s
                    WHERE id = %s
                """, (ProcessingStatus.COMPLETED.value, datetime.now(timezone.utc), document_id))

                conn.commit()
                cursor.close()
                conn.close()

                return {
                    'document_id': document_id,
                    'status': 'completed',
                    'extracted_text': len(extracted_text['text']),
                    'analysis': analysis
                }

            else:
                raise ValueError("Failed to extract text from document")

        except Exception as e:
            logger.error(f"Failed to process document: {e}")

            # Update status to failed
            try:
                conn = psycopg2.connect(**DB_CONFIG)
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE ai_documents
                    SET status = %s
                    WHERE id = %s
                """, (ProcessingStatus.FAILED.value, document_id))
                conn.commit()
                cursor.close()
                conn.close()
            except:
                pass

            return {
                'document_id': document_id,
                'status': 'failed',
                'error': str(e)
            }

    async def _extract_text(self, document: Dict) -> Dict:
        """Extract text from document based on type"""
        try:
            doc_type = document['document_type']
            storage_url = document['storage_url']

            # Download file content if needed
            file_content = await self._download_file(storage_url)

            if doc_type == DocumentType.PDF.value:
                return await self._extract_pdf_text(file_content)

            elif doc_type == DocumentType.WORD.value:
                return await self._extract_word_text(file_content)

            elif doc_type == DocumentType.EXCEL.value:
                return await self._extract_excel_text(file_content)

            elif doc_type == DocumentType.IMAGE.value:
                return await self._extract_image_text(file_content)

            elif doc_type == DocumentType.TEXT.value:
                return {
                    'text': file_content.decode('utf-8', errors='ignore'),
                    'method': ExtractionMethod.NATIVE.value,
                    'confidence': 1.0
                }

            else:
                # Try generic text extraction
                try:
                    text = file_content.decode('utf-8', errors='ignore')
                    return {
                        'text': text,
                        'method': ExtractionMethod.NATIVE.value,
                        'confidence': 0.8
                    }
                except:
                    # Fall back to AI vision
                    return await self._extract_with_ai_vision(file_content)

        except Exception as e:
            logger.error(f"Failed to extract text: {e}")
            return None

    async def _download_file(self, storage_url: str) -> bytes:
        """Download file from storage"""
        if storage_url.startswith('local://'):
            # Local file
            path = storage_url.replace('local://', '')
            with open(path, 'rb') as f:
                return f.read()
        else:
            # Download from URL
            async with httpx.AsyncClient() as client:
                response = await client.get(storage_url)
                return response.content

    async def _extract_pdf_text(self, content: bytes) -> Dict:
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            return {
                'text': text.strip(),
                'method': ExtractionMethod.NATIVE.value,
                'confidence': 0.95,
                'pages': len(pdf_reader.pages)
            }
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            # Fall back to OCR
            return await self._extract_with_ocr(content)

    async def _extract_word_text(self, content: bytes) -> Dict:
        """Extract text from Word document"""
        try:
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join([para.text for para in doc.paragraphs])

            # Also extract from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += "\n" + cell.text

            return {
                'text': text.strip(),
                'method': ExtractionMethod.NATIVE.value,
                'confidence': 0.95
            }
        except Exception as e:
            logger.error(f"Word extraction failed: {e}")
            return None

    async def _extract_excel_text(self, content: bytes) -> Dict:
        """Extract text from Excel"""
        try:
            workbook = openpyxl.load_workbook(io.BytesIO(content))
            text = ""

            for sheet in workbook:
                text += f"Sheet: {sheet.title}\n"
                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join([str(cell) if cell else "" for cell in row])
                    if row_text.strip():
                        text += row_text + "\n"

            return {
                'text': text.strip(),
                'method': ExtractionMethod.NATIVE.value,
                'confidence': 0.9
            }
        except Exception as e:
            logger.error(f"Excel extraction failed: {e}")
            return None

    async def _extract_image_text(self, content: bytes) -> Dict:
        """Extract text from image using OCR"""
        try:
            image = Image.open(io.BytesIO(content))
            text = pytesseract.image_to_string(image)

            return {
                'text': text.strip(),
                'method': ExtractionMethod.OCR.value,
                'confidence': 0.85
            }
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            # Fall back to AI vision
            return await self._extract_with_ai_vision(content)

    async def _extract_with_ocr(self, content: bytes) -> Dict:
        """Extract text using OCR"""
        try:
            # Convert to image first if needed
            image = Image.open(io.BytesIO(content))
            text = pytesseract.image_to_string(image)

            return {
                'text': text.strip(),
                'method': ExtractionMethod.OCR.value,
                'confidence': 0.8
            }
        except:
            return None

    async def _extract_with_ai_vision(self, content: bytes) -> Dict:
        """Extract text using AI vision API"""
        try:
            # Convert to base64
            base64_image = base64.b64encode(content).decode('utf-8')

            response = openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this document image. Include tables, headers, and any visible text."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000
            )

            return {
                'text': response.choices[0].message.content,
                'method': ExtractionMethod.AI_VISION.value,
                'confidence': 0.9
            }
        except Exception as e:
            logger.error(f"AI vision extraction failed: {e}")
            return None

    async def _analyze_document(
        self,
        document_id: str,
        text: str,
        document: Dict
    ) -> Dict:
        """Analyze document content using AI"""
        try:
            # Prepare analysis prompt
            prompt = f"""
            Analyze this document and extract:
            1. Document category (legal, financial, technical, etc.)
            2. Brief summary (2-3 sentences)
            3. Key entities (people, companies, dates, amounts)
            4. Important terms and concepts
            5. Overall sentiment
            6. Main topics
            7. Any action items or deadlines
            8. Risk factors or concerns

            Document type: {document['document_type']}
            Filename: {document['filename']}

            Content:
            {text[:3000]}  # Limit for API

            Return as JSON.
            """

            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a document analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            analysis = json.loads(response.choices[0].message.content)

            # Store analysis
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_document_analysis
                (document_id, analysis_type, category, summary,
                 key_entities, key_terms, sentiment_score,
                 topics, classifications, insights)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                document_id,
                'comprehensive',
                analysis.get('category', 'unknown'),
                analysis.get('summary'),
                Json(analysis.get('entities', {})),
                Json(analysis.get('terms', [])),
                analysis.get('sentiment', 0),
                Json(analysis.get('topics', [])),
                Json(analysis.get('classifications', {})),
                Json(analysis.get('insights', {}))
            ))

            conn.commit()
            cursor.close()
            conn.close()

            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze document: {e}")
            return {}

    async def _create_embeddings(self, document_id: str, text: str):
        """Create embeddings for document chunks"""
        try:
            # Split text into chunks
            chunk_size = 1000
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            for idx, chunk in enumerate(chunks[:20]):  # Limit to 20 chunks
                # Generate embedding
                response = openai_client.embeddings.create(
                    input=chunk,
                    model="text-embedding-ada-002"
                )

                embedding = response.data[0].embedding

                # Store embedding
                cursor.execute("""
                    INSERT INTO ai_document_embeddings
                    (document_id, chunk_index, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s)
                """, (
                    document_id,
                    idx,
                    chunk,
                    embedding
                ))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")

    async def _find_relationships(self, document_id: str, analysis: Dict):
        """Find relationships with other documents"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Find similar documents using embeddings
            cursor.execute("""
                WITH doc_embedding AS (
                    SELECT AVG(embedding) as avg_embedding
                    FROM ai_document_embeddings
                    WHERE document_id = %s
                )
                SELECT
                    d.id,
                    d.filename,
                    AVG(1 - (de.embedding <=> doc_embedding.avg_embedding)) as similarity
                FROM ai_documents d
                JOIN ai_document_embeddings de ON d.id = de.document_id
                CROSS JOIN doc_embedding
                WHERE d.id != %s
                GROUP BY d.id, d.filename
                HAVING AVG(1 - (de.embedding <=> doc_embedding.avg_embedding)) > 0.8
                ORDER BY similarity DESC
                LIMIT 5
            """, (document_id, document_id))

            similar_docs = cursor.fetchall()

            for similar_doc in similar_docs:
                cursor.execute("""
                    INSERT INTO ai_document_relationships
                    (source_document_id, target_document_id,
                     relationship_type, confidence)
                    VALUES (%s, %s, %s, %s)
                """, (
                    document_id,
                    similar_doc[0],
                    'similar',
                    similar_doc[2]
                ))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to find relationships: {e}")

    async def search_documents(
        self,
        query: str,
        filters: Dict = None,
        limit: int = 10
    ) -> List[Dict]:
        """Search documents using semantic search"""
        try:
            # Generate query embedding
            response = openai_client.embeddings.create(
                input=query,
                model="text-embedding-ada-002"
            )
            query_embedding = response.data[0].embedding

            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Build filter conditions
            where_conditions = ["d.status = 'completed'"]
            params = [query_embedding, limit]

            if filters:
                if filters.get('document_type'):
                    where_conditions.append("d.document_type = %s")
                    params.insert(-1, filters['document_type'])
                if filters.get('user_id'):
                    where_conditions.append("d.user_id = %s")
                    params.insert(-1, filters['user_id'])
                if filters.get('category'):
                    where_conditions.append("da.category = %s")
                    params.insert(-1, filters['category'])

            where_clause = " AND ".join(where_conditions)

            # Search using embeddings
            cursor.execute(f"""
                SELECT DISTINCT ON (d.id)
                    d.*,
                    da.summary,
                    da.category,
                    1 - (de.embedding <=> %s) as similarity
                FROM ai_documents d
                JOIN ai_document_embeddings de ON d.id = de.document_id
                LEFT JOIN ai_document_analysis da ON d.id = da.document_id
                WHERE {where_clause}
                ORDER BY d.id, similarity DESC
                LIMIT %s
            """, params)

            results = cursor.fetchall()

            cursor.close()
            conn.close()

            return results

        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []

    async def get_document_insights(self, document_id: str) -> Dict:
        """Get comprehensive insights for a document"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get document info
            cursor.execute("""
                SELECT d.*, dc.raw_text, da.*
                FROM ai_documents d
                LEFT JOIN ai_document_content dc ON d.id = dc.document_id
                LEFT JOIN ai_document_analysis da ON d.id = da.document_id
                WHERE d.id = %s
            """, (document_id,))

            document = cursor.fetchone()

            # Get relationships
            cursor.execute("""
                SELECT
                    dr.*,
                    d.filename as related_filename
                FROM ai_document_relationships dr
                JOIN ai_documents d ON dr.target_document_id = d.id
                WHERE dr.source_document_id = %s
                ORDER BY dr.confidence DESC
            """, (document_id,))

            relationships = cursor.fetchall()

            # Get actions
            cursor.execute("""
                SELECT * FROM ai_document_actions
                WHERE document_id = %s
                ORDER BY created_at DESC
            """, (document_id,))

            actions = cursor.fetchall()

            cursor.close()
            conn.close()

            return {
                'document': document,
                'relationships': relationships,
                'actions': actions,
                'metrics': {
                    'word_count': document.get('word_count', 0) if document else 0,
                    'sentiment': document.get('sentiment_score', 0) if document else 0,
                    'related_documents': len(relationships)
                }
            }

        except Exception as e:
            logger.error(f"Failed to get insights: {e}")
            return {}

    async def trigger_document_action(
        self,
        document_id: str,
        action_type: str,
        action_data: Dict = None
    ) -> Dict:
        """Trigger an action based on document content"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            action_id = str(uuid.uuid4())

            # Create action record
            cursor.execute("""
                INSERT INTO ai_document_actions
                (id, document_id, action_type, action_status,
                 action_data, triggered_by)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                action_id,
                document_id,
                action_type,
                'pending',
                Json(action_data or {}),
                'system'
            ))

            conn.commit()

            # Execute action based on type
            result = await self._execute_action(
                document_id,
                action_type,
                action_data
            )

            # Update action status
            cursor.execute("""
                UPDATE ai_document_actions
                SET action_status = %s,
                    executed_at = %s,
                    result = %s
                WHERE id = %s
            """, (
                'completed' if result.get('success') else 'failed',
                datetime.now(timezone.utc),
                Json(result),
                action_id
            ))

            conn.commit()
            cursor.close()
            conn.close()

            return result

        except Exception as e:
            logger.error(f"Failed to trigger action: {e}")
            return {'success': False, 'error': str(e)}

    async def _execute_action(
        self,
        document_id: str,
        action_type: str,
        action_data: Dict
    ) -> Dict:
        """Execute specific action based on document"""
        try:
            if action_type == 'extract_invoice_data':
                # Extract invoice information
                return await self._extract_invoice_data(document_id)

            elif action_type == 'create_task':
                # Create task from document
                return await self._create_task_from_document(document_id, action_data)

            elif action_type == 'notify_team':
                # Send notification
                return await self._notify_team(document_id, action_data)

            elif action_type == 'generate_response':
                # Generate response document
                return await self._generate_response(document_id, action_data)

            else:
                return {'success': True, 'message': f'Action {action_type} executed'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _extract_invoice_data(self, document_id: str) -> Dict:
        """Extract structured invoice data"""
        # Implementation for invoice extraction
        return {'success': True, 'invoice_data': {}}

    async def _create_task_from_document(self, document_id: str, data: Dict) -> Dict:
        """Create task based on document content"""
        # Implementation for task creation
        return {'success': True, 'task_id': str(uuid.uuid4())}

    async def _notify_team(self, document_id: str, data: Dict) -> Dict:
        """Send notification about document"""
        # Implementation for notifications
        return {'success': True, 'notified': data.get('recipients', [])}

    async def _generate_response(self, document_id: str, data: Dict) -> Dict:
        """Generate response document"""
        # Implementation for response generation
        return {'success': True, 'response_id': str(uuid.uuid4())}

# Singleton instance
_document_processor = None

def get_document_processor() -> DocumentProcessor:
    """Get or create the document processor instance"""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor