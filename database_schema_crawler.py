#!/usr/bin/env python3
"""
Database Schema Crawler
Extracts and documents all database tables, columns, relationships, and foreign keys.
Integrates with the codebase graph for comprehensive system visualization.
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

from config import config
from database.async_connection import PoolConfig, close_pool, get_pool, init_pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Column:
    """Database column metadata"""
    name: str
    data_type: str
    is_nullable: bool
    column_default: Optional[str]
    is_primary_key: bool = False
    is_foreign_key: bool = False
    character_maximum_length: Optional[int] = None


@dataclass
class ForeignKey:
    """Foreign key relationship"""
    constraint_name: str
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    on_delete: Optional[str] = None
    on_update: Optional[str] = None


@dataclass
class Index:
    """Database index metadata"""
    name: str
    table_name: str
    columns: list[str]
    is_unique: bool
    is_primary: bool


@dataclass
class Table:
    """Database table metadata"""
    name: str
    schema_name: str
    columns: list[Column] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    indexes: list[Index] = field(default_factory=list)
    row_count: int = 0
    size_bytes: int = 0
    description: Optional[str] = None


@dataclass
class DatabaseSchema:
    """Complete database schema"""
    tables: list[Table] = field(default_factory=list)
    total_tables: int = 0
    total_columns: int = 0
    total_relationships: int = 0
    crawled_at: str = field(default_factory=lambda: datetime.now().isoformat())


class DatabaseSchemaCrawler:
    """
    Crawls PostgreSQL database schema and extracts comprehensive metadata.
    Stores results for graph visualization and code-to-database mapping.
    """

    def __init__(self):
        self.schema = DatabaseSchema()

    async def run(self) -> DatabaseSchema:
        """Main execution flow"""
        logger.info("Starting database schema crawl...")

        pool = get_pool()

        # 1. Get all tables
        tables = await self._get_tables(pool)
        logger.info(f"Found {len(tables)} tables")

        # 2. For each table, get columns, FKs, indexes
        for table in tables:
            table.columns = await self._get_columns(pool, table.schema_name, table.name)
            table.foreign_keys = await self._get_foreign_keys(pool, table.schema_name, table.name)
            table.indexes = await self._get_indexes(pool, table.schema_name, table.name)
            table.row_count = await self._get_row_count(pool, table.schema_name, table.name)
            table.size_bytes = await self._get_table_size(pool, table.schema_name, table.name)

            # Mark primary keys
            pk_columns = await self._get_primary_key_columns(pool, table.schema_name, table.name)
            for col in table.columns:
                if col.name in pk_columns:
                    col.is_primary_key = True
                if any(fk.source_column == col.name for fk in table.foreign_keys):
                    col.is_foreign_key = True

            self.schema.tables.append(table)

        # 3. Calculate totals
        self.schema.total_tables = len(self.schema.tables)
        self.schema.total_columns = sum(len(t.columns) for t in self.schema.tables)
        self.schema.total_relationships = sum(len(t.foreign_keys) for t in self.schema.tables)

        logger.info(f"Schema crawl complete: {self.schema.total_tables} tables, "
                    f"{self.schema.total_columns} columns, {self.schema.total_relationships} relationships")

        return self.schema

    async def _get_tables(self, pool) -> list[Table]:
        """Get all user tables from database"""
        rows = await pool.fetch("""
            SELECT
                table_schema,
                table_name,
                obj_description((table_schema || '.' || table_name)::regclass, 'pg_class') as description
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                AND table_type = 'BASE TABLE'
            ORDER BY table_schema, table_name
        """)

        return [
            Table(
                name=row['table_name'],
                schema_name=row['table_schema'],
                description=row['description']
            )
            for row in rows
        ]

    async def _get_columns(self, pool, schema: str, table: str) -> list[Column]:
        """Get columns for a table"""
        rows = await pool.fetch("""
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = $1 AND table_name = $2
            ORDER BY ordinal_position
        """, schema, table)

        return [
            Column(
                name=row['column_name'],
                data_type=row['data_type'],
                is_nullable=row['is_nullable'] == 'YES',
                column_default=row['column_default'],
                character_maximum_length=row['character_maximum_length']
            )
            for row in rows
        ]

    async def _get_foreign_keys(self, pool, schema: str, table: str) -> list[ForeignKey]:
        """Get foreign key relationships for a table"""
        rows = await pool.fetch("""
            SELECT
                tc.constraint_name,
                kcu.column_name as source_column,
                ccu.table_name as target_table,
                ccu.column_name as target_column,
                rc.delete_rule as on_delete,
                rc.update_rule as on_update
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            LEFT JOIN information_schema.referential_constraints rc
                ON rc.constraint_name = tc.constraint_name
                AND rc.constraint_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = $1
                AND tc.table_name = $2
        """, schema, table)

        return [
            ForeignKey(
                constraint_name=row['constraint_name'],
                source_table=table,
                source_column=row['source_column'],
                target_table=row['target_table'],
                target_column=row['target_column'],
                on_delete=row['on_delete'],
                on_update=row['on_update']
            )
            for row in rows
        ]

    async def _get_indexes(self, pool, schema: str, table: str) -> list[Index]:
        """Get indexes for a table"""
        rows = await pool.fetch("""
            SELECT
                i.relname as index_name,
                array_agg(a.attname ORDER BY x.n) as columns,
                ix.indisunique as is_unique,
                ix.indisprimary as is_primary
            FROM pg_index ix
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN LATERAL unnest(ix.indkey) WITH ORDINALITY AS x(attnum, n) ON TRUE
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = x.attnum
            WHERE n.nspname = $1 AND t.relname = $2
            GROUP BY i.relname, ix.indisunique, ix.indisprimary
        """, schema, table)

        return [
            Index(
                name=row['index_name'],
                table_name=table,
                columns=row['columns'],
                is_unique=row['is_unique'],
                is_primary=row['is_primary']
            )
            for row in rows
        ]

    async def _get_primary_key_columns(self, pool, schema: str, table: str) -> list[str]:
        """Get primary key columns for a table"""
        rows = await pool.fetch("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
                AND tc.table_schema = $1
                AND tc.table_name = $2
        """, schema, table)

        return [row['column_name'] for row in rows]

    async def _get_row_count(self, pool, schema: str, table: str) -> int:
        """Get approximate row count for a table"""
        try:
            # Use reltuples for fast approximate count
            result = await pool.fetchval("""
                SELECT reltuples::bigint
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = $1 AND c.relname = $2
            """, schema, table)
            return int(result) if result else 0
        except Exception as exc:
            logger.warning("Failed to get row count for %s.%s: %s", schema, table, exc, exc_info=True)
            return 0

    async def _get_table_size(self, pool, schema: str, table: str) -> int:
        """Get table size in bytes"""
        try:
            result = await pool.fetchval("""
                SELECT pg_total_relation_size($1::regclass)
            """, f"{schema}.{table}")
            return int(result) if result else 0
        except Exception as exc:
            logger.warning("Failed to get table size for %s.%s: %s", schema, table, exc, exc_info=True)
            return 0

    async def save_to_graph(self) -> dict[str, Any]:
        """Save schema to codebase graph tables"""
        pool = get_pool()

        nodes_created = 0
        edges_created = 0

        try:
            # Create table nodes
            for table in self.schema.tables:
                # Insert table as a node
                await pool.execute("""
                    INSERT INTO codebase_nodes (name, type, file_path, repo_name, line_number, content_hash, metadata)
                    VALUES ($1, 'database_table', $2, 'database', 0, $3, $4::jsonb)
                    ON CONFLICT (repo_name, file_path, name, type) DO UPDATE
                    SET metadata = EXCLUDED.metadata, updated_at = NOW()
                """,
                    table.name,
                    f"{table.schema_name}.{table.name}",
                    str(hash(json.dumps([c.name for c in table.columns]))),
                    json.dumps({
                        "schema": table.schema_name,
                        "columns": [asdict(c) for c in table.columns],
                        "row_count": table.row_count,
                        "size_bytes": table.size_bytes,
                        "description": table.description
                    })
                )
                nodes_created += 1

            # Create column nodes and relationships
            for table in self.schema.tables:
                for column in table.columns:
                    # Insert column as a node
                    await pool.execute("""
                        INSERT INTO codebase_nodes (name, type, file_path, repo_name, line_number, content_hash, metadata)
                        VALUES ($1, 'database_column', $2, 'database', 0, $3, $4::jsonb)
                        ON CONFLICT (repo_name, file_path, name, type) DO UPDATE
                        SET metadata = EXCLUDED.metadata, updated_at = NOW()
                    """,
                        f"{table.name}.{column.name}",
                        f"{table.schema_name}.{table.name}.{column.name}",
                        str(hash(column.data_type)),
                        json.dumps({
                            "table": table.name,
                            "data_type": column.data_type,
                            "is_nullable": column.is_nullable,
                            "is_primary_key": column.is_primary_key,
                            "is_foreign_key": column.is_foreign_key,
                            "column_default": column.column_default
                        })
                    )
                    nodes_created += 1

            # Create foreign key edges
            for table in self.schema.tables:
                for fk in table.foreign_keys:
                    # Get source and target node IDs
                    source_id = await pool.fetchval("""
                        SELECT id FROM codebase_nodes
                        WHERE name = $1 AND type = 'database_table' AND repo_name = 'database'
                    """, table.name)

                    target_id = await pool.fetchval("""
                        SELECT id FROM codebase_nodes
                        WHERE name = $1 AND type = 'database_table' AND repo_name = 'database'
                    """, fk.target_table)

                    if source_id and target_id:
                        await pool.execute("""
                            INSERT INTO codebase_edges (source_id, target_id, type, metadata)
                            VALUES ($1, $2, 'foreign_key', $3::jsonb)
                            ON CONFLICT (source_id, target_id, type) DO UPDATE
                            SET metadata = EXCLUDED.metadata, updated_at = NOW()
                        """,
                            source_id,
                            target_id,
                            json.dumps({
                                "constraint_name": fk.constraint_name,
                                "source_column": fk.source_column,
                                "target_column": fk.target_column,
                                "on_delete": fk.on_delete,
                                "on_update": fk.on_update
                            })
                        )
                        edges_created += 1

            logger.info(f"Saved to graph: {nodes_created} nodes, {edges_created} edges")

        except Exception as e:
            logger.error(f"Failed to save to graph: {e}")

        return {
            "nodes_created": nodes_created,
            "edges_created": edges_created
        }

    def to_json(self) -> str:
        """Export schema to JSON"""
        return json.dumps({
            "tables": [asdict(t) for t in self.schema.tables],
            "total_tables": self.schema.total_tables,
            "total_columns": self.schema.total_columns,
            "total_relationships": self.schema.total_relationships,
            "crawled_at": self.schema.crawled_at
        }, indent=2, default=str)

    def to_mermaid_erd(self) -> str:
        """Generate Mermaid ERD diagram"""
        lines = ["erDiagram"]

        for table in self.schema.tables:
            # Add table with columns
            lines.append(f"    {table.name} {{")
            for col in table.columns:
                pk = "PK" if col.is_primary_key else ""
                fk = "FK" if col.is_foreign_key else ""
                key_marker = f"{pk}{fk}" if pk or fk else ""
                lines.append(f"        {col.data_type} {col.name} {key_marker}")
            lines.append("    }")

        # Add relationships
        for table in self.schema.tables:
            for fk in table.foreign_keys:
                lines.append(f"    {table.name} ||--o{{ {fk.target_table} : \"{fk.source_column}\"")

        return "\n".join(lines)

    def generate_documentation(self) -> str:
        """Generate markdown documentation"""
        doc = ["# Database Schema Documentation\n"]
        doc.append(f"**Generated:** {self.schema.crawled_at}\n")
        doc.append(f"**Total Tables:** {self.schema.total_tables}")
        doc.append(f"**Total Columns:** {self.schema.total_columns}")
        doc.append(f"**Total Relationships:** {self.schema.total_relationships}\n")

        doc.append("---\n")
        doc.append("## Tables\n")

        for table in sorted(self.schema.tables, key=lambda t: t.name):
            doc.append(f"### {table.schema_name}.{table.name}")
            if table.description:
                doc.append(f"_{table.description}_\n")

            doc.append(f"- **Rows:** ~{table.row_count:,}")
            doc.append(f"- **Size:** {self._format_bytes(table.size_bytes)}\n")

            # Columns table
            doc.append("| Column | Type | Nullable | Default | Key |")
            doc.append("|--------|------|----------|---------|-----|")
            for col in table.columns:
                nullable = "✓" if col.is_nullable else ""
                key = ""
                if col.is_primary_key:
                    key = "PK"
                elif col.is_foreign_key:
                    key = "FK"
                default = col.column_default[:20] + "..." if col.column_default and len(col.column_default) > 20 else (col.column_default or "")
                doc.append(f"| {col.name} | {col.data_type} | {nullable} | {default} | {key} |")

            # Foreign keys
            if table.foreign_keys:
                doc.append("\n**Foreign Keys:**")
                for fk in table.foreign_keys:
                    doc.append(f"- `{fk.source_column}` → `{fk.target_table}.{fk.target_column}` ({fk.constraint_name})")

            doc.append("")

        # ERD
        doc.append("---\n")
        doc.append("## Entity Relationship Diagram\n")
        doc.append("```mermaid")
        doc.append(self.to_mermaid_erd())
        doc.append("```\n")

        return "\n".join(doc)

    def _format_bytes(self, size: int) -> str:
        """Format bytes to human readable"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


async def main():
    """Main entry point"""
    # Initialize database pool
    db_config = config.database
    pool_config = PoolConfig(
        host=db_config.host,
        port=db_config.port,
        user=db_config.user,
        password=db_config.password,
        database=db_config.database,
        ssl=db_config.ssl,
        ssl_verify=db_config.ssl_verify
    )

    await init_pool(pool_config)

    try:
        crawler = DatabaseSchemaCrawler()

        # Crawl schema
        schema = await crawler.run()

        # Save to graph
        graph_result = await crawler.save_to_graph()
        print(f"\nSaved to codebase graph: {graph_result}")

        # Generate documentation
        doc = crawler.generate_documentation()
        doc_path = "/home/matt-woodworth/dev/DATABASE_SCHEMA.md"
        with open(doc_path, 'w') as f:
            f.write(doc)
        print(f"\nDocumentation saved to: {doc_path}")

        # Export JSON
        json_path = "/home/matt-woodworth/dev/brainops-ai-agents/database_schema.json"
        with open(json_path, 'w') as f:
            f.write(crawler.to_json())
        print(f"JSON schema saved to: {json_path}")

        # Print summary
        print("\n=== Database Schema Summary ===")
        print(f"Tables: {schema.total_tables}")
        print(f"Columns: {schema.total_columns}")
        print(f"Foreign Keys: {schema.total_relationships}")

        # Print largest tables
        largest = sorted(schema.tables, key=lambda t: t.row_count, reverse=True)[:10]
        print("\nLargest Tables:")
        for t in largest:
            print(f"  {t.name}: ~{t.row_count:,} rows ({crawler._format_bytes(t.size_bytes)})")

    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
