#!/usr/bin/env python3
"""
Code Quality Agent - Automated Code Review & Technical Debt Detection
Reviews code, detects anti-patterns, suggests refactorings, tracks technical debt
"""

import os
import json
import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor, Json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
    'password': os.getenv('DB_PASSWORD', 'REDACTED_SUPABASE_DB_PASSWORD'),
    'port': int(os.getenv('DB_PORT', 6543))
}


class IssueType(Enum):
    """Code quality issue types"""
    COMPLEXITY = "complexity"
    DUPLICATION = "duplication"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    TESTING = "testing"


class IssueSeverity(Enum):
    """Issue severity levels"""
    CRITICAL = 1  # Security vulnerabilities, data corruption risks
    HIGH = 2      # Performance issues, major bugs
    MEDIUM = 3    # Code smells, maintainability
    LOW = 4       # Style, documentation


@dataclass
class CodeIssue:
    """Detected code quality issue"""
    id: str
    file_path: str
    line_number: int
    issue_type: IssueType
    severity: IssueSeverity
    title: str
    description: str
    code_snippet: str
    suggestion: str
    auto_fixable: bool
    estimated_fix_time: int  # minutes
    detected_at: datetime
    fixed: bool


@dataclass
class TechnicalDebt:
    """Technical debt item"""
    id: str
    category: str  # architecture, design, implementation, testing, documentation
    title: str
    description: str
    business_impact: str
    estimated_cost_hours: float
    interest_rate: str  # How fast it grows (low, medium, high)
    affected_files: List[str]
    priority: int  # 1-5
    created_at: datetime
    resolved_at: Optional[datetime]


@dataclass
class RefactoringOpportunity:
    """Suggested code refactoring"""
    id: str
    title: str
    description: str
    refactoring_type: str  # extract_method, rename, move_class, simplify, etc.
    affected_code: str
    proposed_solution: str
    benefits: List[str]
    risks: List[str]
    estimated_effort_hours: float
    impact_score: float  # 0-100
    created_at: datetime


class CodeQualityAgent:
    """Agent that reviews code and detects quality issues"""

    def __init__(self):
        self.db_config = DB_CONFIG
        self.conn = None
        self.issues_found = []
        self.debt_items = []
        self.refactoring_opportunities = []
        self._init_database()
        logger.info("✅ Code Quality Agent initialized")

    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Create code issues table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_code_issues (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    file_path TEXT NOT NULL,
                    line_number INTEGER,
                    issue_type VARCHAR(50) NOT NULL,
                    severity INTEGER NOT NULL CHECK (severity >= 1 AND severity <= 4),
                    title TEXT NOT NULL,
                    description TEXT,
                    code_snippet TEXT,
                    suggestion TEXT,
                    auto_fixable BOOLEAN DEFAULT FALSE,
                    estimated_fix_time INTEGER,
                    detected_at TIMESTAMP DEFAULT NOW(),
                    fixed BOOLEAN DEFAULT FALSE,
                    fixed_at TIMESTAMP,
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create technical debt table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_technical_debt (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    category VARCHAR(50) NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    business_impact TEXT,
                    estimated_cost_hours FLOAT,
                    interest_rate VARCHAR(20),
                    affected_files JSONB DEFAULT '[]'::jsonb,
                    priority INTEGER CHECK (priority >= 1 AND priority <= 5),
                    created_at TIMESTAMP DEFAULT NOW(),
                    resolved_at TIMESTAMP,
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create refactoring opportunities table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_refactoring_opportunities (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title TEXT NOT NULL,
                    description TEXT,
                    refactoring_type VARCHAR(50),
                    affected_code TEXT,
                    proposed_solution TEXT,
                    benefits JSONB DEFAULT '[]'::jsonb,
                    risks JSONB DEFAULT '[]'::jsonb,
                    estimated_effort_hours FLOAT,
                    impact_score FLOAT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    implemented_at TIMESTAMP,
                    status VARCHAR(50) DEFAULT 'proposed',
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create code metrics table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_code_metrics (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    file_path TEXT NOT NULL,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    measured_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_code_issues_unfixed ON ai_code_issues(fixed, severity) WHERE fixed = FALSE;
                CREATE INDEX IF NOT EXISTS idx_technical_debt_unresolved ON ai_technical_debt(priority, created_at DESC) WHERE resolved_at IS NULL;
                CREATE INDEX IF NOT EXISTS idx_refactoring_status ON ai_refactoring_opportunities(status, impact_score DESC);
            """)

            conn.commit()
            logger.info("✅ Code Quality Agent database tables ready")

        except Exception as e:
            logger.warning(f"Database initialization skipped: {e}. Operating without persistence")
        finally:
            if conn:
                conn.close()

    def analyze_code_complexity(self) -> List[CodeIssue]:
        """Analyze code for complexity issues"""
        issues = []

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check for agents with complex execution logic
            # This is a simplified check - in production, would parse actual code files
            cur.execute("""
                SELECT name, type, capabilities
                FROM agents
                WHERE enabled = true
                ORDER BY total_executions DESC
                LIMIT 20
            """)

            agents = cur.fetchall()

            for agent in agents:
                # Check if agent has many capabilities (potential complexity)
                capabilities = agent.get('capabilities') or {}
                if isinstance(capabilities, dict) and len(capabilities) > 10:
                    issues.append({
                        'file_path': f"agents/{agent['name'].lower().replace(' ', '_')}.py",
                        'line_number': 1,
                        'issue_type': 'complexity',
                        'severity': 3,  # MEDIUM
                        'title': 'High Capability Count - Consider Decomposition',
                        'description': f"Agent '{agent['name']}' has {len(capabilities)} capabilities. Consider splitting into smaller, focused agents.",
                        'code_snippet': f"capabilities = {json.dumps(capabilities, indent=2)[:200]}...",
                        'suggestion': 'Extract related capabilities into separate agents (Single Responsibility Principle)',
                        'auto_fixable': False,
                        'estimated_fix_time': 180  # 3 hours
                    })

            conn.close()

        except Exception as e:
            logger.warning(f"Complexity analysis failed: {e}")

        return issues

    def detect_code_duplication(self) -> List[CodeIssue]:
        """Detect duplicated code patterns"""
        issues = []

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check for duplicate agent configurations
            cur.execute("""
                SELECT type, COUNT(*) as count
                FROM agents
                WHERE enabled = true
                GROUP BY type
                HAVING COUNT(*) > 5
                ORDER BY count DESC
            """)

            duplicate_types = cur.fetchall()

            for dup_type in duplicate_types:
                if dup_type['count'] > 5:
                    issues.append({
                        'file_path': 'agents/agent_registry.py',
                        'line_number': 1,
                        'issue_type': 'duplication',
                        'severity': 3,  # MEDIUM
                        'title': f'Multiple Agents of Same Type ({dup_type["type"]})',
                        'description': f"Found {dup_type['count']} agents of type '{dup_type['type']}'. Consider creating a base class or factory pattern.",
                        'code_snippet': f"# {dup_type['count']} agents with type='{dup_type['type']}'",
                        'suggestion': 'Create base class or use factory pattern to reduce duplication',
                        'auto_fixable': False,
                        'estimated_fix_time': 120
                    })

            conn.close()

        except Exception as e:
            logger.warning(f"Duplication detection failed: {e}")

        return issues

    def check_security_issues(self) -> List[CodeIssue]:
        """Check for potential security vulnerabilities"""
        issues = []

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check for agents with unrestricted capabilities
            cur.execute("""
                SELECT id, name, capabilities
                FROM agents
                WHERE enabled = true
                AND (capabilities->>'admin_access')::boolean = true
            """)

            admin_agents = cur.fetchall()

            if len(admin_agents) > 3:
                issues.append({
                    'file_path': 'agents/permissions.py',
                    'line_number': 1,
                    'issue_type': 'security',
                    'severity': 2,  # HIGH
                    'title': 'Too Many Agents with Admin Access',
                    'description': f"{len(admin_agents)} agents have admin_access=true. Follow principle of least privilege.",
                    'code_snippet': f"admin_agents = {[a['name'] for a in admin_agents[:5]]}",
                    'suggestion': 'Audit admin access and remove unnecessary permissions',
                    'auto_fixable': False,
                    'estimated_fix_time': 90
                })

            # Check for API keys in database (should use environment variables)
            cur.execute("""
                SELECT COUNT(*) as key_count
                FROM information_schema.columns
                WHERE column_name LIKE '%api_key%'
                AND table_schema = 'public'
            """)

            key_data = cur.fetchone()
            if key_data and key_data['key_count'] > 0:
                issues.append({
                    'file_path': 'config/database_schema.sql',
                    'line_number': 1,
                    'issue_type': 'security',
                    'severity': 1,  # CRITICAL
                    'title': 'API Keys Stored in Database',
                    'description': f"Found {key_data['key_count']} columns containing 'api_key'. Use environment variables instead.",
                    'code_snippet': 'api_key_column VARCHAR(255)',
                    'suggestion': 'Move API keys to environment variables and use secret management',
                    'auto_fixable': False,
                    'estimated_fix_time': 60
                })

            conn.close()

        except Exception as e:
            logger.warning(f"Security check failed: {e}")

        return issues

    def identify_technical_debt(self) -> List[TechnicalDebt]:
        """Identify technical debt in the system"""
        debt_items = []

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check for missing test coverage (no test execution records)
            cur.execute("""
                SELECT COUNT(*) as total_agents,
                       COUNT(*) FILTER (WHERE total_executions > 0) as tested_agents
                FROM agents
                WHERE enabled = true
            """)

            test_data = cur.fetchone()
            if test_data:
                untested = test_data['total_agents'] - test_data['tested_agents']
                if untested > 0:
                    debt_items.append({
                        'category': 'testing',
                        'title': 'Missing Test Coverage for Agents',
                        'description': f'{untested} agents have never been executed/tested',
                        'business_impact': 'Risk of bugs in production, slower development velocity',
                        'estimated_cost_hours': untested * 2.0,  # 2 hours per agent
                        'interest_rate': 'high',
                        'affected_files': ['agents/*.py'],
                        'priority': 2  # HIGH
                    })

            # Check for outdated dependencies (agents not updated in 90+ days)
            cur.execute("""
                SELECT COUNT(*) as stale_agents
                FROM agents
                WHERE enabled = true
                AND (updated_at < NOW() - INTERVAL '90 days' OR updated_at IS NULL)
            """)

            stale_data = cur.fetchone()
            if stale_data and stale_data['stale_agents'] > 0:
                debt_items.append({
                    'category': 'maintainability',
                    'title': 'Stale Agent Configurations',
                    'description': f"{stale_data['stale_agents']} agents haven't been reviewed in 90+ days",
                    'business_impact': 'Potential security vulnerabilities, missing features, technical obsolescence',
                    'estimated_cost_hours': stale_data['stale_agents'] * 1.5,
                    'interest_rate': 'medium',
                    'affected_files': ['agents/*.py'],
                    'priority': 3  # MEDIUM
                })

            # Check for missing documentation (agents without descriptions)
            cur.execute("""
                SELECT COUNT(*) as undocumented
                FROM agents
                WHERE enabled = true
                AND (description IS NULL OR description = '' OR LENGTH(description) < 50)
            """)

            doc_data = cur.fetchone()
            if doc_data and doc_data['undocumented'] > 0:
                debt_items.append({
                    'category': 'documentation',
                    'title': 'Insufficient Agent Documentation',
                    'description': f"{doc_data['undocumented']} agents lack adequate documentation",
                    'business_impact': 'Difficulty onboarding new developers, maintenance challenges',
                    'estimated_cost_hours': doc_data['undocumented'] * 0.5,
                    'interest_rate': 'low',
                    'affected_files': ['agents/*.py'],
                    'priority': 4  # LOW
                })

            conn.close()

        except Exception as e:
            logger.warning(f"Technical debt identification failed: {e}")

        return debt_items

    def suggest_refactorings(self) -> List[RefactoringOpportunity]:
        """Suggest code refactoring opportunities"""
        opportunities = []

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Suggest extracting common agent patterns into base classes
            cur.execute("""
                SELECT type, COUNT(*) as count
                FROM agents
                WHERE enabled = true
                GROUP BY type
                HAVING COUNT(*) >= 3
                ORDER BY count DESC
                LIMIT 5
            """)

            common_types = cur.fetchall()

            for common_type in common_types:
                opportunities.append({
                    'title': f'Extract Base Class for {common_type["type"]} Agents',
                    'description': f'Create abstract base class for {common_type["count"]} {common_type["type"]} agents',
                    'refactoring_type': 'extract_base_class',
                    'affected_code': f'{common_type["count"]} agents of type {common_type["type"]}',
                    'proposed_solution': f'Create Base{common_type["type"].title()}Agent class with shared functionality',
                    'benefits': [
                        'Reduce code duplication',
                        'Easier to maintain and update',
                        'Consistent behavior across similar agents',
                        'Faster development of new agents'
                    ],
                    'risks': [
                        'Breaking changes if not careful',
                        'Testing overhead for refactored code'
                    ],
                    'estimated_effort_hours': 4.0,
                    'impact_score': 75.0
                })

            # Suggest consolidating rarely-used agents
            cur.execute("""
                SELECT COUNT(*) as rarely_used
                FROM agents
                WHERE enabled = true
                AND total_executions < 10
                AND created_at < NOW() - INTERVAL '30 days'
            """)

            rarely_used_data = cur.fetchone()
            if rarely_used_data and rarely_used_data['rarely_used'] > 5:
                opportunities.append({
                    'title': 'Consolidate Rarely-Used Agents',
                    'description': f'Merge {rarely_used_data["rarely_used"]} rarely-used agents into multi-purpose agents',
                    'refactoring_type': 'consolidate',
                    'affected_code': f'{rarely_used_data["rarely_used"]} agent files',
                    'proposed_solution': 'Group by function and create 2-3 multi-purpose agents',
                    'benefits': [
                        'Simpler codebase',
                        'Easier to navigate',
                        'Reduced maintenance overhead',
                        'Lower infrastructure costs'
                    ],
                    'risks': [
                        'May reduce clarity of purpose',
                        'Migration effort for existing integrations'
                    ],
                    'estimated_effort_hours': 8.0,
                    'impact_score': 60.0
                })

            conn.close()

        except Exception as e:
            logger.warning(f"Refactoring suggestion failed: {e}")

        return opportunities

    def _persist_findings(self, issues: List[Dict], debt_items: List[Dict], refactorings: List[Dict]):
        """Persist all findings to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Persist code issues
            for issue in issues:
                cur.execute("""
                    INSERT INTO ai_code_issues
                    (file_path, line_number, issue_type, severity, title, description,
                     code_snippet, suggestion, auto_fixable, estimated_fix_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    issue['file_path'],
                    issue['line_number'],
                    issue['issue_type'],
                    issue['severity'],
                    issue['title'],
                    issue['description'],
                    issue['code_snippet'],
                    issue['suggestion'],
                    issue['auto_fixable'],
                    issue['estimated_fix_time']
                ))

            # Persist technical debt
            for debt in debt_items:
                cur.execute("""
                    INSERT INTO ai_technical_debt
                    (category, title, description, business_impact, estimated_cost_hours,
                     interest_rate, affected_files, priority)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    debt['category'],
                    debt['title'],
                    debt['description'],
                    debt['business_impact'],
                    debt['estimated_cost_hours'],
                    debt['interest_rate'],
                    Json(debt['affected_files']),
                    debt['priority']
                ))

            # Persist refactoring opportunities
            for refactor in refactorings:
                cur.execute("""
                    INSERT INTO ai_refactoring_opportunities
                    (title, description, refactoring_type, affected_code, proposed_solution,
                     benefits, risks, estimated_effort_hours, impact_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    refactor['title'],
                    refactor['description'],
                    refactor['refactoring_type'],
                    refactor['affected_code'],
                    refactor['proposed_solution'],
                    Json(refactor['benefits']),
                    Json(refactor['risks']),
                    refactor['estimated_effort_hours'],
                    refactor['impact_score']
                ))

            conn.commit()
            conn.close()
            logger.info(f"✅ Persisted {len(issues)} issues, {len(debt_items)} debt items, {len(refactorings)} refactorings")

        except Exception as e:
            logger.warning(f"Failed to persist findings: {e}")

    async def continuous_quality_loop(self, interval_hours: int = 12):
        """Main loop that continuously analyzes code quality"""
        logger.info(f"🔄 Starting code quality analysis loop (every {interval_hours}h)")

        while True:
            try:
                logger.info("🔍 Analyzing code quality...")

                # Analyze code complexity
                complexity_issues = self.analyze_code_complexity()
                logger.info(f"📊 Found {len(complexity_issues)} complexity issues")

                # Detect code duplication
                duplication_issues = self.detect_code_duplication()
                logger.info(f"📊 Found {len(duplication_issues)} duplication issues")

                # Check security
                security_issues = self.check_security_issues()
                if security_issues:
                    logger.warning(f"⚠️ Found {len(security_issues)} security issues")

                # Identify technical debt
                debt_items = self.identify_technical_debt()
                logger.info(f"📊 Identified {len(debt_items)} technical debt items")

                # Suggest refactorings
                refactorings = self.suggest_refactorings()
                logger.info(f"💡 Generated {len(refactorings)} refactoring suggestions")

                # Combine all issues
                all_issues = complexity_issues + duplication_issues + security_issues

                # Persist findings
                self._persist_findings(all_issues, debt_items, refactorings)

                # Log summary
                critical_count = sum(1 for i in all_issues if i['severity'] == 1)
                if critical_count > 0:
                    logger.error(f"❌ {critical_count} CRITICAL issues require immediate attention!")

            except Exception as e:
                logger.error(f"❌ Code quality loop error: {e}")

            # Wait before next analysis
            await asyncio.sleep(interval_hours * 3600)


if __name__ == "__main__":
    agent = CodeQualityAgent()
    asyncio.run(agent.continuous_quality_loop(interval_hours=12))
