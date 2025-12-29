#!/usr/bin/env python3
"""
Data Pipeline Automation System - Task 17
Automated data pipeline management for ETL, data transformation, and workflow orchestration
"""

import os
import json
import logging
import asyncio
import uuid
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal
import psycopg2
from psycopg2.extras import RealDictCursor, Json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT", "5432")
}


class PipelineType(Enum):
    """Types of data pipelines"""
    ETL = "etl"
    ELT = "elt"
    STREAMING = "streaming"
    BATCH = "batch"
    REAL_TIME = "real_time"
    INCREMENTAL = "incremental"
    FULL_REFRESH = "full_refresh"
    CDC = "cdc"  # Change Data Capture


class PipelineStatus(Enum):
    """Status of pipeline execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    WAITING = "waiting"


class TransformationType(Enum):
    """Types of data transformations"""
    FILTER = "filter"
    MAP = "map"
    AGGREGATE = "aggregate"
    JOIN = "join"
    DEDUPLICATE = "deduplicate"
    NORMALIZE = "normalize"
    ENRICH = "enrich"
    VALIDATE = "validate"
    CUSTOM = "custom"


class DataSourceType(Enum):
    """Types of data sources"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    S3 = "s3"
    API = "api"
    CSV = "csv"
    JSON = "json"
    KAFKA = "kafka"
    WEBHOOK = "webhook"


class ScheduleFrequency(Enum):
    """Pipeline schedule frequencies"""
    ONCE = "once"
    MINUTELY = "minutely"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CRON = "cron"


@dataclass
class DataSource:
    """Data source configuration"""
    source_id: str
    source_type: DataSourceType
    connection_config: Dict
    name: str
    description: Optional[str] = None
    schema: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class TransformationStep:
    """A single transformation step in the pipeline"""
    step_id: str
    transformation_type: TransformationType
    config: Dict
    order: int
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict] = None
    output_schema: Optional[Dict] = None


@dataclass
class PipelineConfig:
    """Configuration for a data pipeline"""
    pipeline_id: str
    name: str
    pipeline_type: PipelineType
    sources: List[DataSource]
    transformations: List[TransformationStep]
    destination: DataSource
    schedule: Optional[Dict] = None
    retry_config: Optional[Dict] = None
    alerts: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class PipelineRun:
    """Record of a pipeline execution"""
    run_id: str
    pipeline_id: str
    status: PipelineStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    records_processed: int = 0
    records_failed: int = 0
    error_message: Optional[str] = None
    step_results: List[Dict] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


class DataExtractor:
    """Extract data from various sources"""

    def __init__(self):
        self.extractors: Dict[DataSourceType, Callable] = {
            DataSourceType.POSTGRESQL: self._extract_postgresql,
            DataSourceType.API: self._extract_api,
            DataSourceType.CSV: self._extract_csv,
            DataSourceType.JSON: self._extract_json,
        }

    async def extract(
        self,
        source: DataSource,
        incremental_key: Optional[str] = None,
        last_value: Optional[Any] = None
    ) -> List[Dict]:
        """Extract data from a source"""
        extractor = self.extractors.get(source.source_type)
        if not extractor:
            raise ValueError(f"Unsupported source type: {source.source_type}")

        return await extractor(source, incremental_key, last_value)

    async def _extract_postgresql(
        self,
        source: DataSource,
        incremental_key: Optional[str],
        last_value: Optional[Any]
    ) -> List[Dict]:
        """Extract data from PostgreSQL"""
        try:
            config = source.connection_config
            conn = psycopg2.connect(
                host=config.get('host', DB_CONFIG['host']),
                database=config.get('database', DB_CONFIG['database']),
                user=config.get('user', DB_CONFIG['user']),
                password=config.get('password', DB_CONFIG['password']),
                port=config.get('port', DB_CONFIG['port'])
            )
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            query = config.get('query', f"SELECT * FROM {config.get('table', 'unknown')}")

            if incremental_key and last_value:
                query += f" WHERE {incremental_key} > %s"
                cursor.execute(query, (last_value,))
            else:
                cursor.execute(query)

            data = [dict(row) for row in cursor.fetchall()]

            cursor.close()
            conn.close()

            return data

        except Exception as e:
            logger.error(f"PostgreSQL extraction error: {e}")
            raise

    async def _extract_api(
        self,
        source: DataSource,
        incremental_key: Optional[str],
        last_value: Optional[Any]
    ) -> List[Dict]:
        """Extract data from API"""
        try:
            import httpx

            config = source.connection_config
            url = config.get('url')
            method = config.get('method', 'GET')
            headers = config.get('headers', {})
            params = config.get('params', {})

            if incremental_key and last_value:
                params[incremental_key] = last_value

            async with httpx.AsyncClient() as client:
                if method.upper() == 'GET':
                    response = await client.get(url, headers=headers, params=params)
                else:
                    response = await client.post(
                        url, headers=headers, params=params,
                        json=config.get('body', {})
                    )

                response.raise_for_status()
                data = response.json()

                # Handle nested data
                data_path = config.get('data_path')
                if data_path:
                    for key in data_path.split('.'):
                        data = data.get(key, [])

                return data if isinstance(data, list) else [data]

        except Exception as e:
            logger.error(f"API extraction error: {e}")
            raise

    async def _extract_csv(
        self,
        source: DataSource,
        incremental_key: Optional[str],
        last_value: Optional[Any]
    ) -> List[Dict]:
        """Extract data from CSV"""
        try:
            import csv
            config = source.connection_config
            file_path = config.get('file_path')

            data = []
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if incremental_key and last_value:
                        if row.get(incremental_key, '') > str(last_value):
                            data.append(dict(row))
                    else:
                        data.append(dict(row))

            return data

        except Exception as e:
            logger.error(f"CSV extraction error: {e}")
            raise

    async def _extract_json(
        self,
        source: DataSource,
        incremental_key: Optional[str],
        last_value: Optional[Any]
    ) -> List[Dict]:
        """Extract data from JSON file"""
        try:
            config = source.connection_config
            file_path = config.get('file_path')

            with open(file_path, 'r') as f:
                data = json.load(f)

            if not isinstance(data, list):
                data = [data]

            if incremental_key and last_value:
                data = [d for d in data if d.get(incremental_key, '') > str(last_value)]

            return data

        except Exception as e:
            logger.error(f"JSON extraction error: {e}")
            raise


class DataTransformer:
    """Transform data through pipeline steps"""

    def __init__(self):
        self.transformers: Dict[TransformationType, Callable] = {
            TransformationType.FILTER: self._filter,
            TransformationType.MAP: self._map,
            TransformationType.AGGREGATE: self._aggregate,
            TransformationType.DEDUPLICATE: self._deduplicate,
            TransformationType.NORMALIZE: self._normalize,
            TransformationType.ENRICH: self._enrich,
            TransformationType.VALIDATE: self._validate,
        }

    async def transform(
        self,
        data: List[Dict],
        step: TransformationStep
    ) -> List[Dict]:
        """Apply a transformation step to data"""
        transformer = self.transformers.get(step.transformation_type)
        if not transformer:
            if step.transformation_type == TransformationType.CUSTOM:
                return await self._custom_transform(data, step.config)
            raise ValueError(f"Unsupported transformation: {step.transformation_type}")

        return await transformer(data, step.config)

    async def _filter(self, data: List[Dict], config: Dict) -> List[Dict]:
        """Filter data based on conditions"""
        field = config.get('field')
        operator = config.get('operator', 'eq')
        value = config.get('value')

        operators = {
            'eq': lambda x, v: x == v,
            'ne': lambda x, v: x != v,
            'gt': lambda x, v: x > v,
            'gte': lambda x, v: x >= v,
            'lt': lambda x, v: x < v,
            'lte': lambda x, v: x <= v,
            'in': lambda x, v: x in v,
            'contains': lambda x, v: v in str(x),
            'is_null': lambda x, v: x is None,
            'not_null': lambda x, v: x is not None,
        }

        op_func = operators.get(operator, operators['eq'])
        return [d for d in data if op_func(d.get(field), value)]

    async def _map(self, data: List[Dict], config: Dict) -> List[Dict]:
        """Map/transform field values"""
        mappings = config.get('mappings', {})
        result = []

        for record in data:
            new_record = record.copy()
            for source_field, target_config in mappings.items():
                if isinstance(target_config, str):
                    # Simple rename
                    if source_field in new_record:
                        new_record[target_config] = new_record.pop(source_field)
                elif isinstance(target_config, dict):
                    # Transform with function
                    value = new_record.get(source_field)
                    transform_type = target_config.get('transform')
                    target_field = target_config.get('target', source_field)

                    if transform_type == 'uppercase':
                        new_record[target_field] = str(value).upper() if value else None
                    elif transform_type == 'lowercase':
                        new_record[target_field] = str(value).lower() if value else None
                    elif transform_type == 'to_int':
                        new_record[target_field] = int(value) if value else None
                    elif transform_type == 'to_float':
                        new_record[target_field] = float(value) if value else None
                    elif transform_type == 'to_string':
                        new_record[target_field] = str(value) if value else None

            result.append(new_record)

        return result

    async def _aggregate(self, data: List[Dict], config: Dict) -> List[Dict]:
        """Aggregate data"""
        group_by = config.get('group_by', [])
        aggregations = config.get('aggregations', {})

        if not group_by:
            # Aggregate all data
            result = {}
            for agg_field, agg_config in aggregations.items():
                values = [d.get(agg_config['field']) for d in data if d.get(agg_config['field']) is not None]
                result[agg_field] = self._calculate_aggregate(values, agg_config['function'])
            return [result]

        # Group by fields
        groups = {}
        for record in data:
            key = tuple(record.get(f) for f in group_by)
            if key not in groups:
                groups[key] = []
            groups[key].append(record)

        results = []
        for key, group_data in groups.items():
            result = dict(zip(group_by, key))
            for agg_field, agg_config in aggregations.items():
                values = [d.get(agg_config['field']) for d in group_data if d.get(agg_config['field']) is not None]
                result[agg_field] = self._calculate_aggregate(values, agg_config['function'])
            results.append(result)

        return results

    def _calculate_aggregate(self, values: List[Any], function: str) -> Any:
        """Calculate aggregate value"""
        if not values:
            return None

        numeric_values = [float(v) for v in values if v is not None]

        if function == 'sum':
            return sum(numeric_values)
        elif function == 'avg':
            return sum(numeric_values) / len(numeric_values) if numeric_values else None
        elif function == 'min':
            return min(numeric_values) if numeric_values else None
        elif function == 'max':
            return max(numeric_values) if numeric_values else None
        elif function == 'count':
            return len(values)
        elif function == 'count_distinct':
            return len(set(values))
        else:
            return None

    async def _deduplicate(self, data: List[Dict], config: Dict) -> List[Dict]:
        """Remove duplicate records"""
        key_fields = config.get('key_fields', [])
        keep = config.get('keep', 'first')  # first or last

        if not key_fields:
            # Use entire record as key
            seen = set()
            result = []
            for record in data:
                record_hash = hashlib.md5(
                    json.dumps(record, sort_keys=True, default=str).encode()
                ).hexdigest()
                if record_hash not in seen:
                    seen.add(record_hash)
                    result.append(record)
            return result

        seen = {}
        for record in data:
            key = tuple(record.get(f) for f in key_fields)
            if keep == 'first' and key not in seen:
                seen[key] = record
            elif keep == 'last':
                seen[key] = record

        return list(seen.values())

    async def _normalize(self, data: List[Dict], config: Dict) -> List[Dict]:
        """Normalize data values"""
        fields = config.get('fields', {})
        result = []

        for record in data:
            new_record = record.copy()
            for field, norm_config in fields.items():
                value = new_record.get(field)
                if value is not None:
                    norm_type = norm_config.get('type')
                    if norm_type == 'trim':
                        new_record[field] = str(value).strip()
                    elif norm_type == 'phone':
                        # Simple phone normalization
                        new_record[field] = ''.join(c for c in str(value) if c.isdigit())
                    elif norm_type == 'email':
                        new_record[field] = str(value).lower().strip()
                    elif norm_type == 'date':
                        # Could add date format normalization
                        pass
            result.append(new_record)

        return result

    async def _enrich(self, data: List[Dict], config: Dict) -> List[Dict]:
        """Enrich data with additional fields"""
        enrichments = config.get('enrichments', [])
        result = []

        for record in data:
            new_record = record.copy()
            for enrichment in enrichments:
                field = enrichment.get('field')
                enrichment_type = enrichment.get('type')

                if enrichment_type == 'timestamp':
                    new_record[field] = datetime.now(timezone.utc).isoformat()
                elif enrichment_type == 'uuid':
                    new_record[field] = str(uuid.uuid4())
                elif enrichment_type == 'static':
                    new_record[field] = enrichment.get('value')
                elif enrichment_type == 'computed':
                    # Simple expression evaluation
                    expression = enrichment.get('expression', '')
                    # Could add safe expression evaluation here

            result.append(new_record)

        return result

    async def _validate(self, data: List[Dict], config: Dict) -> List[Dict]:
        """Validate data and filter invalid records"""
        rules = config.get('rules', [])
        valid_records = []

        for record in data:
            is_valid = True
            for rule in rules:
                field = rule.get('field')
                validation = rule.get('validation')
                value = record.get(field)

                if validation == 'required' and value is None:
                    is_valid = False
                elif validation == 'not_empty' and not value:
                    is_valid = False
                elif validation == 'numeric' and value is not None:
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        is_valid = False
                elif validation == 'email' and value:
                    if '@' not in str(value):
                        is_valid = False

                if not is_valid:
                    break

            if is_valid:
                valid_records.append(record)

        return valid_records

    async def _custom_transform(self, data: List[Dict], config: Dict) -> List[Dict]:
        """Apply custom transformation"""
        # Placeholder for custom transformation logic
        logger.info("Custom transformation applied")
        return data


class DataLoader:
    """Load data to destination"""

    def __init__(self):
        self.loaders: Dict[DataSourceType, Callable] = {
            DataSourceType.POSTGRESQL: self._load_postgresql,
            DataSourceType.API: self._load_api,
        }

    async def load(
        self,
        data: List[Dict],
        destination: DataSource,
        mode: str = "append"  # append, replace, upsert
    ) -> Dict:
        """Load data to destination"""
        loader = self.loaders.get(destination.source_type)
        if not loader:
            raise ValueError(f"Unsupported destination type: {destination.source_type}")

        return await loader(data, destination, mode)

    async def _load_postgresql(
        self,
        data: List[Dict],
        destination: DataSource,
        mode: str
    ) -> Dict:
        """Load data to PostgreSQL"""
        if not data:
            return {"records_loaded": 0}

        try:
            config = destination.connection_config
            conn = psycopg2.connect(
                host=config.get('host', DB_CONFIG['host']),
                database=config.get('database', DB_CONFIG['database']),
                user=config.get('user', DB_CONFIG['user']),
                password=config.get('password', DB_CONFIG['password']),
                port=config.get('port', DB_CONFIG['port'])
            )
            cursor = conn.cursor()

            table = config.get('table')
            columns = list(data[0].keys())

            if mode == "replace":
                cursor.execute(f"TRUNCATE TABLE {table}")

            if mode == "upsert":
                key_column = config.get('key_column', 'id')
                for record in data:
                    values = [record.get(c) for c in columns]
                    placeholders = ', '.join(['%s'] * len(columns))
                    update_set = ', '.join([
                        f"{c} = EXCLUDED.{c}" for c in columns if c != key_column
                    ])
                    cursor.execute(f"""
                        INSERT INTO {table} ({', '.join(columns)})
                        VALUES ({placeholders})
                        ON CONFLICT ({key_column}) DO UPDATE SET {update_set}
                    """, values)
            else:
                for record in data:
                    values = [record.get(c) for c in columns]
                    placeholders = ', '.join(['%s'] * len(columns))
                    cursor.execute(f"""
                        INSERT INTO {table} ({', '.join(columns)})
                        VALUES ({placeholders})
                    """, values)

            conn.commit()
            cursor.close()
            conn.close()

            return {"records_loaded": len(data)}

        except Exception as e:
            logger.error(f"PostgreSQL load error: {e}")
            raise

    async def _load_api(
        self,
        data: List[Dict],
        destination: DataSource,
        mode: str
    ) -> Dict:
        """Load data via API"""
        try:
            import httpx

            config = destination.connection_config
            url = config.get('url')
            headers = config.get('headers', {})
            batch_size = config.get('batch_size', 100)

            records_loaded = 0
            async with httpx.AsyncClient() as client:
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    response = await client.post(
                        url, headers=headers, json=batch
                    )
                    response.raise_for_status()
                    records_loaded += len(batch)

            return {"records_loaded": records_loaded}

        except Exception as e:
            logger.error(f"API load error: {e}")
            raise


class PipelineScheduler:
    """Schedule and manage pipeline executions"""

    def __init__(self):
        self.schedules: Dict[str, Dict] = {}
        self._running = False

    async def schedule_pipeline(
        self,
        pipeline_id: str,
        frequency: ScheduleFrequency,
        config: Optional[Dict] = None
    ) -> str:
        """Schedule a pipeline for execution"""
        schedule_id = str(uuid.uuid4())
        self.schedules[schedule_id] = {
            "pipeline_id": pipeline_id,
            "frequency": frequency,
            "config": config or {},
            "next_run": self._calculate_next_run(frequency, config),
            "enabled": True
        }

        # Store in database
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_pipeline_schedules
                (id, pipeline_id, frequency, config, next_run, enabled)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                schedule_id,
                pipeline_id,
                frequency.value,
                Json(config or {}),
                self.schedules[schedule_id]["next_run"],
                True
            ))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to save schedule: {e}")

        return schedule_id

    def _calculate_next_run(
        self,
        frequency: ScheduleFrequency,
        config: Optional[Dict]
    ) -> datetime:
        """Calculate next run time"""
        now = datetime.now(timezone.utc)

        if frequency == ScheduleFrequency.ONCE:
            return config.get('run_at', now) if config else now
        elif frequency == ScheduleFrequency.MINUTELY:
            return now + timedelta(minutes=1)
        elif frequency == ScheduleFrequency.HOURLY:
            return now + timedelta(hours=1)
        elif frequency == ScheduleFrequency.DAILY:
            run_time = config.get('time', '00:00') if config else '00:00'
            hour, minute = map(int, run_time.split(':'))
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run
        elif frequency == ScheduleFrequency.WEEKLY:
            return now + timedelta(weeks=1)
        elif frequency == ScheduleFrequency.MONTHLY:
            return now + timedelta(days=30)
        else:
            return now + timedelta(hours=1)


class DataPipelineAutomation:
    """Main data pipeline automation system"""

    def __init__(self):
        self.extractor = DataExtractor()
        self.transformer = DataTransformer()
        self.loader = DataLoader()
        self.scheduler = PipelineScheduler()
        self.conn = None
        self._init_database()

    def _get_connection(self):
        """Get database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**DB_CONFIG)
        return self.conn

    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Pipelines table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_data_pipelines (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    pipeline_type VARCHAR(50),
                    sources JSONB DEFAULT '[]',
                    transformations JSONB DEFAULT '[]',
                    destination JSONB DEFAULT '{}',
                    schedule JSONB DEFAULT '{}',
                    retry_config JSONB DEFAULT '{}',
                    alerts JSONB DEFAULT '{}',
                    status VARCHAR(50) DEFAULT 'active',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Pipeline runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_pipeline_runs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    pipeline_id UUID REFERENCES ai_data_pipelines(id),
                    status VARCHAR(50),
                    started_at TIMESTAMPTZ DEFAULT NOW(),
                    completed_at TIMESTAMPTZ,
                    records_processed INT DEFAULT 0,
                    records_failed INT DEFAULT 0,
                    error_message TEXT,
                    step_results JSONB DEFAULT '[]',
                    metrics JSONB DEFAULT '{}'
                )
            """)

            # Pipeline schedules table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_pipeline_schedules (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    pipeline_id UUID REFERENCES ai_data_pipelines(id),
                    frequency VARCHAR(50),
                    config JSONB DEFAULT '{}',
                    next_run TIMESTAMPTZ,
                    last_run TIMESTAMPTZ,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Pipeline metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_pipeline_metrics (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    pipeline_id UUID REFERENCES ai_data_pipelines(id),
                    run_id UUID REFERENCES ai_pipeline_runs(id),
                    metric_name VARCHAR(100),
                    metric_value FLOAT,
                    recorded_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status
                ON ai_pipeline_runs(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pipeline_runs_pipeline
                ON ai_pipeline_runs(pipeline_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pipeline_schedules_next
                ON ai_pipeline_schedules(next_run)
            """)

            conn.commit()
            cursor.close()

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    async def create_pipeline(
        self,
        name: str,
        pipeline_type: PipelineType,
        sources: List[Dict],
        transformations: List[Dict],
        destination: Dict,
        schedule: Optional[Dict] = None,
        retry_config: Optional[Dict] = None,
        alerts: Optional[Dict] = None
    ) -> str:
        """Create a new data pipeline"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            pipeline_id = str(uuid.uuid4())

            cursor.execute("""
                INSERT INTO ai_data_pipelines
                (id, name, pipeline_type, sources, transformations,
                 destination, schedule, retry_config, alerts)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                pipeline_id,
                name,
                pipeline_type.value,
                Json(sources),
                Json(transformations),
                Json(destination),
                Json(schedule or {}),
                Json(retry_config or {"max_retries": 3, "retry_delay": 60}),
                Json(alerts or {})
            ))

            conn.commit()
            cursor.close()

            logger.info(f"Created pipeline: {pipeline_id}")
            return pipeline_id

        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            raise

    async def run_pipeline(
        self,
        pipeline_id: str,
        parameters: Optional[Dict] = None
    ) -> PipelineRun:
        """Execute a data pipeline"""
        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)

        try:
            # Get pipeline config
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT * FROM ai_data_pipelines WHERE id = %s
            """, (pipeline_id,))

            pipeline = cursor.fetchone()
            if not pipeline:
                raise ValueError(f"Pipeline not found: {pipeline_id}")

            # Create run record
            cursor.execute("""
                INSERT INTO ai_pipeline_runs
                (id, pipeline_id, status, started_at)
                VALUES (%s, %s, %s, %s)
            """, (run_id, pipeline_id, PipelineStatus.RUNNING.value, started_at))
            conn.commit()

            # Extract data from sources
            all_data = []
            step_results = []

            for source_config in pipeline['sources']:
                source = DataSource(
                    source_id=source_config.get('id', str(uuid.uuid4())),
                    source_type=DataSourceType(source_config['type']),
                    connection_config=source_config.get('connection', {}),
                    name=source_config.get('name', 'Unknown')
                )

                extracted = await self.extractor.extract(
                    source,
                    parameters.get('incremental_key') if parameters else None,
                    parameters.get('last_value') if parameters else None
                )

                all_data.extend(extracted)
                step_results.append({
                    "step": "extract",
                    "source": source.name,
                    "records": len(extracted)
                })

            # Apply transformations
            transformed_data = all_data
            for trans_config in pipeline['transformations']:
                step = TransformationStep(
                    step_id=trans_config.get('id', str(uuid.uuid4())),
                    transformation_type=TransformationType(trans_config['type']),
                    config=trans_config.get('config', {}),
                    order=trans_config.get('order', 0),
                    name=trans_config.get('name', 'Unknown')
                )

                transformed_data = await self.transformer.transform(
                    transformed_data, step
                )

                step_results.append({
                    "step": "transform",
                    "transformation": step.name,
                    "records": len(transformed_data)
                })

            # Load to destination
            dest_config = pipeline['destination']
            destination = DataSource(
                source_id=dest_config.get('id', str(uuid.uuid4())),
                source_type=DataSourceType(dest_config['type']),
                connection_config=dest_config.get('connection', {}),
                name=dest_config.get('name', 'Unknown')
            )

            load_result = await self.loader.load(
                transformed_data,
                destination,
                dest_config.get('mode', 'append')
            )

            step_results.append({
                "step": "load",
                "destination": destination.name,
                "records": load_result['records_loaded']
            })

            # Update run status
            completed_at = datetime.now(timezone.utc)
            cursor.execute("""
                UPDATE ai_pipeline_runs
                SET status = %s, completed_at = %s,
                    records_processed = %s, step_results = %s,
                    metrics = %s
                WHERE id = %s
            """, (
                PipelineStatus.COMPLETED.value,
                completed_at,
                load_result['records_loaded'],
                Json(step_results),
                Json({
                    "duration_seconds": (completed_at - started_at).total_seconds(),
                    "records_per_second": load_result['records_loaded'] /
                        max((completed_at - started_at).total_seconds(), 1)
                }),
                run_id
            ))
            conn.commit()
            cursor.close()

            return PipelineRun(
                run_id=run_id,
                pipeline_id=pipeline_id,
                status=PipelineStatus.COMPLETED,
                started_at=started_at,
                completed_at=completed_at,
                records_processed=load_result['records_loaded'],
                step_results=step_results
            )

        except Exception as e:
            logger.error(f"Pipeline run failed: {e}")

            # Update run status
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE ai_pipeline_runs
                    SET status = %s, error_message = %s, completed_at = NOW()
                    WHERE id = %s
                """, (PipelineStatus.FAILED.value, str(e), run_id))
                conn.commit()
                cursor.close()
            except Exception:
                pass

            return PipelineRun(
                run_id=run_id,
                pipeline_id=pipeline_id,
                status=PipelineStatus.FAILED,
                started_at=started_at,
                error_message=str(e)
            )

    async def get_pipeline_status(self, pipeline_id: str) -> Dict:
        """Get status of a pipeline"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT p.*,
                    (SELECT COUNT(*) FROM ai_pipeline_runs WHERE pipeline_id = p.id) as total_runs,
                    (SELECT COUNT(*) FROM ai_pipeline_runs WHERE pipeline_id = p.id AND status = 'completed') as successful_runs
                FROM ai_data_pipelines p
                WHERE p.id = %s
            """, (pipeline_id,))

            pipeline = cursor.fetchone()

            # Get recent runs
            cursor.execute("""
                SELECT * FROM ai_pipeline_runs
                WHERE pipeline_id = %s
                ORDER BY started_at DESC
                LIMIT 10
            """, (pipeline_id,))

            recent_runs = cursor.fetchall()

            cursor.close()

            return {
                "pipeline": dict(pipeline) if pipeline else None,
                "recent_runs": [dict(r) for r in recent_runs],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            return {"error": str(e)}

    async def list_pipelines(
        self,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """List all pipelines"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            query = "SELECT * FROM ai_data_pipelines"
            params = []

            if status:
                query += " WHERE status = %s"
                params.append(status)

            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)

            cursor.execute(query, params)
            pipelines = cursor.fetchall()
            cursor.close()

            return [dict(p) for p in pipelines]

        except Exception as e:
            logger.error(f"Failed to list pipelines: {e}")
            return []


# Singleton instance
_pipeline_system: Optional[DataPipelineAutomation] = None


def get_data_pipeline_automation():
    """Get or create the data pipeline automation instance"""
    global _pipeline_system
    if _pipeline_system is None:
        _pipeline_system = DataPipelineAutomation()
    return _pipeline_system


# Export main components
__all__ = [
    'DataPipelineAutomation',
    'get_data_pipeline_automation',
    'PipelineType',
    'PipelineStatus',
    'TransformationType',
    'DataSourceType',
    'ScheduleFrequency',
    'PipelineConfig',
    'PipelineRun'
]
