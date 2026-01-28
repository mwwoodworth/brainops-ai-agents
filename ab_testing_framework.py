#!/usr/bin/env python3
"""
A/B Testing Framework - Task 24
Test and optimize AI decisions with real experiments
"""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration - validate required environment variables
def _get_db_config():
    """Get database configuration with validation for required env vars."""
    required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        database_url = os.getenv('DATABASE_URL', '')
        if database_url:
            parsed = urlparse(database_url)
            return {
                'host': parsed.hostname or '',
                'database': parsed.path.lstrip('/') if parsed.path else 'postgres',
                'user': parsed.username or '',
                'password': parsed.password or '',
                'port': int(str(parsed.port)) if parsed.port else 5432
            }
        raise RuntimeError("Missing required: DB_HOST/DB_USER/DB_PASSWORD or DATABASE_URL")

    return {
        "host": os.getenv("DB_HOST"),
        "database": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": int(os.getenv("DB_PORT", "5432"))
    }

class ExperimentStatus(Enum):
    """Experiment status"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class VariantType(Enum):
    """Types of variants"""
    CONTROL = "control"
    TREATMENT = "treatment"

class MetricType(Enum):
    """Types of metrics to track"""
    CONVERSION = "conversion"
    REVENUE = "revenue"
    ENGAGEMENT = "engagement"
    RETENTION = "retention"
    CUSTOM = "custom"

class StatisticalSignificance(Enum):
    """Statistical significance levels"""
    NOT_SIGNIFICANT = "not_significant"
    LOW = "low"              # p < 0.10
    MEDIUM = "medium"        # p < 0.05
    HIGH = "high"           # p < 0.01
    VERY_HIGH = "very_high" # p < 0.001

class ExperimentDesigner:
    """Design and configure A/B experiments"""

    def __init__(self):
        self.min_sample_size = 100
        self.confidence_level = 0.95
        self.power = 0.80

    async def create_experiment(
        self,
        name: str,
        hypothesis: str,
        variants: list[dict],
        metrics: list[dict],
        allocation: Optional[dict] = None,
        duration_days: int = 14
    ) -> str:
        """Create a new A/B test experiment"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor()

            experiment_id = str(uuid.uuid4())

            # Default allocation if not specified
            if not allocation:
                allocation = self._calculate_allocation(len(variants))

            # Store experiment
            cursor.execute("""
                INSERT INTO ab_experiments (
                    id, name, hypothesis, status,
                    variants, metrics, allocation,
                    start_date, end_date, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                experiment_id,
                name,
                hypothesis,
                ExperimentStatus.DRAFT.value,
                json.dumps(variants),
                json.dumps(metrics),
                json.dumps(allocation),
                datetime.now(timezone.utc),
                datetime.now(timezone.utc) + timedelta(days=duration_days)
            ))

            # Create variant records
            for variant in variants:
                variant_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO ab_variants (
                        id, experiment_id, name, type,
                        configuration, allocation_percentage
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    variant_id,
                    experiment_id,
                    variant['name'],
                    variant.get('type', VariantType.TREATMENT.value),
                    json.dumps(variant.get('config', {})),
                    allocation.get(variant['name'], 50)
                ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Created experiment {experiment_id}: {name}")
            return experiment_id

        except Exception as e:
            logger.error(f"Error creating experiment: {e!r}")
            raise

    def _calculate_allocation(self, num_variants: int) -> dict:
        """Calculate equal allocation for variants"""
        if num_variants == 2:
            return {"control": 50, "treatment": 50}
        else:
            allocation = 100 / num_variants
            return {f"variant_{i}": allocation for i in range(num_variants)}

    async def calculate_sample_size(
        self,
        baseline_rate: float,
        minimum_effect: float,
        alpha: float = 0.05,
        power: float = 0.80
    ) -> int:
        """Calculate required sample size for statistical significance"""
        from scipy.stats import norm

        # Z-scores for alpha and power
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)

        # Calculate sample size
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_effect)
        p_avg = (p1 + p2) / 2

        n = (2 * p_avg * (1 - p_avg) * (z_alpha + z_beta)**2) / (p1 - p2)**2

        return int(np.ceil(n))

class TrafficSplitter:
    """Split traffic between variants"""

    def __init__(self):
        self.assignment_cache = {}

    async def assign_variant(
        self,
        experiment_id: str,
        user_id: str
    ) -> str:
        """Assign user to a variant"""
        try:
            # Check cache first
            cache_key = f"{experiment_id}:{user_id}"
            if cache_key in self.assignment_cache:
                return self.assignment_cache[cache_key]

            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Check existing assignment
            cursor.execute("""
                SELECT variant_name
                FROM ab_assignments
                WHERE experiment_id = %s AND user_id = %s
            """, (experiment_id, user_id))

            existing = cursor.fetchone()
            if existing:
                existing_variant = existing.get('variant_name') or 'control'
                self.assignment_cache[cache_key] = existing_variant
                cursor.close()
                conn.close()
                return existing_variant

            # Get experiment allocation
            cursor.execute("""
                SELECT allocation, variants
                FROM ab_experiments
                WHERE id = %s AND status = 'running'
            """, (experiment_id,))

            experiment = cursor.fetchone()
            if not experiment:
                cursor.close()
                conn.close()
                return "control"  # Default to control if experiment not running

            # Assign based on hash
            allocation = experiment.get('allocation') or {}
            assignment = self._hash_assignment(
                user_id,
                allocation
            )

            # Store assignment
            cursor.execute("""
                INSERT INTO ab_assignments (
                    experiment_id, user_id, variant_name,
                    assigned_at
                ) VALUES (%s, %s, %s, NOW())
            """, (experiment_id, user_id, assignment))

            conn.commit()
            cursor.close()
            conn.close()

            self.assignment_cache[cache_key] = assignment
            return assignment

        except Exception as e:
            logger.error(f"Error assigning variant: {e!r}")
            return "control"  # Default to control on error

    def _hash_assignment(
        self,
        user_id: str,
        allocation: dict
    ) -> str:
        """Hash-based variant assignment"""
        if not allocation:
            return "control"
        if isinstance(allocation, str):
            try:
                allocation = json.loads(allocation)
            except (TypeError, json.JSONDecodeError):
                return "control"
        # Create hash
        hash_input = f"{user_id}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)

        # Map to percentage
        percentage = (hash_value % 100) + 1

        # Assign based on allocation
        cumulative = 0
        for variant, alloc_pct in allocation.items():
            cumulative += alloc_pct
            if percentage <= cumulative:
                return variant

        return list(allocation.keys())[0]  # Default to first variant

class MetricsCollector:
    """Collect and track experiment metrics"""

    def __init__(self):
        self.metric_buffer = []

    async def track_event(
        self,
        experiment_id: str,
        user_id: str,
        event_type: str,
        value: Optional[float] = None,
        metadata: Optional[dict] = None
    ) -> bool:
        """Track an event for an experiment"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get user's variant
            cursor.execute("""
                SELECT variant_name
                FROM ab_assignments
                WHERE experiment_id = %s AND user_id = %s
            """, (experiment_id, user_id))

            assignment = cursor.fetchone()
            if not assignment:
                cursor.close()
                conn.close()
                return False

            # Record event
            event_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO ab_events (
                    id, experiment_id, user_id, variant_name,
                    event_type, value, metadata, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                event_id,
                experiment_id,
                user_id,
                assignment['variant_name'],
                event_type,
                value,
                json.dumps(metadata or {})
            ))

            conn.commit()
            cursor.close()
            conn.close()

            return True

        except Exception as e:
            logger.error(f"Error tracking event: {e!r}")
            return False

    async def calculate_metrics(
        self,
        experiment_id: str
    ) -> dict:
        """Calculate current metrics for experiment"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get variant metrics
            cursor.execute("""
                WITH variant_stats AS (
                    SELECT
                        variant_name,
                        COUNT(DISTINCT user_id) as users,
                        COUNT(*) as events,
                        AVG(value) as avg_value,
                        SUM(value) as total_value,
                        COUNT(CASE WHEN event_type = 'conversion' THEN 1 END) as conversions
                    FROM ab_events
                    WHERE experiment_id = %s
                    GROUP BY variant_name
                )
                SELECT
                    variant_name,
                    users,
                    events,
                    COALESCE(avg_value, 0) as avg_value,
                    COALESCE(total_value, 0) as total_value,
                    conversions,
                    CASE WHEN users > 0
                        THEN (conversions::float / users) * 100
                        ELSE 0
                    END as conversion_rate
                FROM variant_stats
            """, (experiment_id,))

            metrics = cursor.fetchall()

            cursor.close()
            conn.close()

            return {
                "variants": [dict(m) for m in metrics],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {e!r}")
            raise

class StatisticalAnalyzer:
    """Analyze experiment results for statistical significance"""

    def __init__(self):
        self.min_observations = 30

    async def analyze_experiment(
        self,
        experiment_id: str
    ) -> dict:
        """Perform statistical analysis on experiment results"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get experiment data
            cursor.execute("""
                SELECT
                    variant_name,
                    COUNT(DISTINCT user_id) as users,
                    COUNT(CASE WHEN event_type = 'conversion' THEN 1 END) as conversions,
                    AVG(value) as mean_value,
                    STDDEV(value) as std_value
                FROM ab_events
                WHERE experiment_id = %s
                GROUP BY variant_name
                HAVING COUNT(DISTINCT user_id) >= %s
            """, (experiment_id, self.min_observations))

            results = cursor.fetchall()

            if len(results) < 2:
                cursor.close()
                conn.close()
                return {
                    "status": "insufficient_data",
                    "message": f"Need at least {self.min_observations} users per variant"
                }

            # Perform statistical tests
            control = next(
                (r for r in results if 'control' in (r.get('variant_name') or '').lower()),
                results[0]
            )
            treatments = [r for r in results if r != control]

            analysis = {
                "control": self._format_variant_stats(control),
                "treatments": [],
                "winner": None,
                "confidence": 0,
                "recommendation": None
            }

            for treatment in treatments:
                comparison = self._compare_variants(control, treatment)
                analysis["treatments"].append(comparison)

                # Check for winner
                if comparison['p_value'] < 0.05 and comparison['lift'] > 0:
                    if not analysis["winner"] or comparison['lift'] > analysis["winner"]["lift"]:
                        analysis["winner"] = {
                            "variant": treatment['variant_name'],
                            "lift": comparison['lift'],
                            "confidence": (1 - comparison['p_value']) * 100
                        }

            # Generate recommendation
            analysis["recommendation"] = self._generate_recommendation(analysis)

            cursor.close()
            conn.close()

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing experiment: {e!r}")
            raise

    def _format_variant_stats(self, variant: dict) -> dict:
        """Format variant statistics"""
        users = variant.get('users') or 0
        conversions = variant.get('conversions') or 0
        mean_value = variant.get('mean_value') or 0
        std_value = variant.get('std_value') or 0

        return {
            "name": variant.get('variant_name') or 'unknown',
            "users": int(users or 0),
            "conversions": int(conversions or 0),
            "conversion_rate": (conversions / users * 100) if users > 0 else 0,
            "mean_value": float(mean_value or 0),
            "std_value": float(std_value or 0)
        }

    def _compare_variants(
        self,
        control: dict,
        treatment: dict
    ) -> dict:
        """Compare treatment to control"""
        # Conversion rate comparison
        control_users = control.get('users') or 0
        control_conversions = control.get('conversions') or 0
        treatment_users = treatment.get('users') or 0
        treatment_conversions = treatment.get('conversions') or 0

        control_rate = control_conversions / control_users if control_users > 0 else 0
        treatment_rate = treatment_conversions / treatment_users if treatment_users > 0 else 0

        # Calculate lift
        lift = ((treatment_rate - control_rate) / control_rate * 100) if control_rate > 0 else 0

        # Perform chi-square test
        from scipy.stats import chi2_contingency

        contingency_table = [
            [control_conversions, control_users - control_conversions],
            [treatment_conversions, treatment_users - treatment_conversions]
        ]

        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # Determine significance level
        significance = self._determine_significance(p_value)

        return {
            "variant": treatment['variant_name'],
            "lift": round(lift, 2),
            "control_rate": round(control_rate * 100, 2),
            "treatment_rate": round(treatment_rate * 100, 2),
            "p_value": round(p_value, 4),
            "significance": significance.value,
            "is_significant": p_value < 0.05
        }

    def _determine_significance(self, p_value: float) -> StatisticalSignificance:
        """Determine statistical significance level"""
        if p_value < 0.001:
            return StatisticalSignificance.VERY_HIGH
        elif p_value < 0.01:
            return StatisticalSignificance.HIGH
        elif p_value < 0.05:
            return StatisticalSignificance.MEDIUM
        elif p_value < 0.10:
            return StatisticalSignificance.LOW
        else:
            return StatisticalSignificance.NOT_SIGNIFICANT

    def _generate_recommendation(self, analysis: dict) -> str:
        """Generate recommendation based on analysis"""
        if analysis["winner"]:
            winner = analysis["winner"]
            return f"Deploy {winner['variant']} - {winner['lift']:.1f}% lift with {winner['confidence']:.1f}% confidence"
        else:
            return "Continue experiment - no significant winner yet"

class ExperimentManager:
    """Manage experiment lifecycle"""

    def __init__(self):
        self.active_experiments = {}

    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE ab_experiments
                SET status = %s, start_date = NOW()
                WHERE id = %s AND status = 'draft'
            """, (ExperimentStatus.RUNNING.value, experiment_id))

            affected = cursor.rowcount
            conn.commit()
            cursor.close()
            conn.close()

            if affected > 0:
                logger.info(f"Started experiment {experiment_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Error starting experiment: {e!r}")
            return False

    async def stop_experiment(self, experiment_id: str) -> dict:
        """Stop an experiment and get final results"""
        try:
            # Get final analysis
            analyzer = StatisticalAnalyzer()
            final_results = await analyzer.analyze_experiment(experiment_id)

            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor()

            # Update experiment status
            cursor.execute("""
                UPDATE ab_experiments
                SET
                    status = %s,
                    end_date = NOW(),
                    results = %s
                WHERE id = %s
            """, (
                ExperimentStatus.COMPLETED.value,
                json.dumps(final_results),
                experiment_id
            ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Completed experiment {experiment_id}")
            return final_results

        except Exception as e:
            logger.error(f"Error stopping experiment: {e!r}")
            raise

    async def get_experiment_status(self, experiment_id: str) -> dict:
        """Get current experiment status and metrics"""
        try:
            collector = MetricsCollector()
            analyzer = StatisticalAnalyzer()

            metrics = await collector.calculate_metrics(experiment_id)
            analysis = await analyzer.analyze_experiment(experiment_id)

            return {
                "experiment_id": experiment_id,
                "metrics": metrics,
                "analysis": analysis,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting experiment status: {e!r}")
            raise

class ABTestingFramework:
    """Main A/B testing framework"""

    def __init__(self):
        self.designer = ExperimentDesigner()
        self.splitter = TrafficSplitter()
        self.collector = MetricsCollector()
        self.analyzer = StatisticalAnalyzer()
        self.manager = ExperimentManager()

    async def run_experiment(
        self,
        name: str,
        hypothesis: str,
        control_config: dict,
        treatment_config: dict,
        duration_days: int = 14
    ) -> str:
        """Run a complete A/B test"""
        # Create experiment
        experiment_id = await self.designer.create_experiment(
            name=name,
            hypothesis=hypothesis,
            variants=[
                {"name": "control", "type": "control", "config": control_config},
                {"name": "treatment", "type": "treatment", "config": treatment_config}
            ],
            metrics=[{"type": "conversion", "name": "primary_conversion"}],
            duration_days=duration_days
        )

        # Start experiment
        await self.manager.start_experiment(experiment_id)

        return experiment_id

    async def get_variant(
        self,
        experiment_id: str,
        user_id: str
    ) -> str:
        """Get variant assignment for user"""
        return await self.splitter.assign_variant(experiment_id, user_id)

    async def track(
        self,
        experiment_id: str,
        user_id: str,
        event: str,
        value: Optional[float] = None
    ) -> bool:
        """Track user event"""
        return await self.collector.track_event(
            experiment_id,
            user_id,
            event,
            value
        )

    async def analyze(self, experiment_id: str) -> dict:
        """Analyze experiment results"""
        return await self.analyzer.analyze_experiment(experiment_id)

    async def complete(self, experiment_id: str) -> dict:
        """Complete experiment and get results"""
        return await self.manager.stop_experiment(experiment_id)

# Singleton instance
_framework_instance = None

def get_ab_testing_framework():
    """Get or create A/B testing framework instance"""
    global _framework_instance
    if _framework_instance is None:
        _framework_instance = ABTestingFramework()
    return _framework_instance

# Export main components
__all__ = [
    'ABTestingFramework',
    'get_ab_testing_framework',
    'ExperimentStatus',
    'MetricType',
    'StatisticalSignificance'
]
