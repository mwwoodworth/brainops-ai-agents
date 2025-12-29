"""
AI Decision Tree for Autonomous Operations
Provides decision-making framework for independent agent actions
"""

import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, asdict, field
import os
from dotenv import load_dotenv
import asyncio
import logging
import uuid
import random
from collections import defaultdict

load_dotenv()
logger = logging.getLogger(__name__)

class DecisionType(Enum):
    STRATEGIC = "strategic"          # Long-term business decisions
    OPERATIONAL = "operational"      # Day-to-day operations
    TACTICAL = "tactical"            # Mid-term optimizations
    EMERGENCY = "emergency"          # Crisis response
    FINANCIAL = "financial"          # Revenue/cost decisions
    CUSTOMER = "customer"            # Customer-facing decisions
    TECHNICAL = "technical"          # Technical/infrastructure
    LEARNING = "learning"            # Self-improvement decisions

class ActionType(Enum):
    EXECUTE = "execute"              # Direct action execution
    DELEGATE = "delegate"            # Delegate to another agent
    MONITOR = "monitor"              # Monitor and wait
    ESCALATE = "escalate"            # Escalate to human
    PARALLEL = "parallel"            # Execute multiple actions
    SEQUENTIAL = "sequential"        # Execute actions in order
    CONDITIONAL = "conditional"      # Execute based on conditions
    RETRY = "retry"                  # Retry failed action

class ConfidenceLevel(Enum):
    HIGH = "high"                    # >80% confidence
    MEDIUM = "medium"                # 50-80% confidence
    LOW = "low"                      # <50% confidence
    UNCERTAIN = "uncertain"          # Need more data

@dataclass
class DecisionContext:
    """Context information for decision making"""
    current_state: Dict[str, Any]
    historical_data: List[Dict]
    constraints: Dict[str, Any]
    objectives: List[str]
    available_resources: Dict[str, float]
    time_constraints: Optional[timedelta]
    risk_tolerance: float = 0.5
    metadata: Dict = field(default_factory=dict)

@dataclass
class DecisionOption:
    """Represents a potential decision option"""
    option_id: str
    action: ActionType
    description: str
    confidence: float
    expected_outcome: Dict[str, Any]
    risks: List[str]
    requirements: List[str]
    estimated_duration: timedelta
    cost_estimate: float
    success_probability: float
    impact_score: float

@dataclass
class DecisionNode:
    """Node in the decision tree"""
    node_id: str
    node_type: DecisionType
    question: str
    options: List[DecisionOption]
    parent_id: Optional[str]
    children: List[str]
    evaluation_criteria: Dict[str, float]
    threshold: float
    metadata: Dict

@dataclass
class RiskAssessment:
    """Risk assessment for a decision"""
    overall_risk_score: float  # 0-1 (0=no risk, 1=critical risk)
    risk_categories: Dict[str, float]  # category -> score
    mitigation_strategies: List[str]
    escalation_required: bool
    human_review_required: bool
    risk_factors: List[str]

@dataclass
class MultiCriteriaScore:
    """Multi-criteria decision analysis result"""
    criteria_scores: Dict[str, float]  # criterion -> normalized score (0-1)
    weighted_scores: Dict[str, float]  # criterion -> weighted score
    total_score: float
    sensitivity_analysis: Dict[str, Any]
    pareto_efficient: bool

@dataclass
class DecisionAuditEntry:
    """Audit trail entry for a decision"""
    entry_id: str
    decision_id: str
    timestamp: datetime
    event_type: str  # 'created', 'evaluated', 'executed', 'outcome_recorded', 'learned'
    actor: str  # 'system', 'human', 'agent_name'
    action: str
    context: Dict[str, Any]
    changes: Dict[str, Any]

@dataclass
class DecisionResult:
    """Result of a decision process"""
    decision_id: str
    timestamp: datetime
    selected_option: DecisionOption
    confidence_level: ConfidenceLevel
    reasoning: str
    alternative_options: List[DecisionOption]
    execution_plan: List[Dict]
    monitoring_plan: Dict
    success_criteria: List[str]
    rollback_plan: Optional[Dict]
    # Enhanced fields
    multi_criteria_analysis: Optional[MultiCriteriaScore] = None
    risk_assessment: Optional[RiskAssessment] = None
    audit_trail: List[DecisionAuditEntry] = field(default_factory=list)
    human_escalation_triggered: bool = False
    escalation_reason: Optional[str] = None

class AIDecisionTree:
    """AI Decision Tree for autonomous decision making"""

    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
            'database': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
            'password': os.getenv("DB_PASSWORD"),
            'port': os.getenv('DB_PORT', 5432)
        }
        self.decision_trees = {}
        self.active_decisions = {}
        self.decision_history = []
        self.learning_buffer = []
        self._learning_tables_initialized = False
        self._initialize_database()
        self._load_decision_trees()
        self._register_decision_handlers()

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def _initialize_database(self):
        """Initialize database tables for decision tree"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Create decision trees table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_decision_trees (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tree_name VARCHAR(255) UNIQUE NOT NULL,
                    tree_type VARCHAR(50) NOT NULL,
                    root_node_id VARCHAR(255),
                    description TEXT,
                    version INT DEFAULT 1,
                    is_active BOOLEAN DEFAULT true,
                    performance_score FLOAT DEFAULT 0.0,
                    usage_count INT DEFAULT 0,
                    success_rate FLOAT DEFAULT 0.0,
                    tree_structure JSONB NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create decision nodes table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_decision_nodes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    node_id VARCHAR(255) UNIQUE NOT NULL,
                    tree_id UUID REFERENCES ai_decision_trees(id),
                    node_type VARCHAR(50) NOT NULL,
                    parent_id VARCHAR(255),
                    question TEXT NOT NULL,
                    evaluation_function TEXT,
                    threshold FLOAT DEFAULT 0.5,
                    options JSONB DEFAULT '[]'::jsonb,
                    children TEXT[] DEFAULT '{}',
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create decision history table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_decision_history (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    decision_id VARCHAR(255) UNIQUE NOT NULL,
                    tree_id UUID REFERENCES ai_decision_trees(id),
                    node_path TEXT[] NOT NULL,
                    context JSONB NOT NULL,
                    selected_option JSONB NOT NULL,
                    confidence_level VARCHAR(20),
                    reasoning TEXT,
                    execution_plan JSONB DEFAULT '[]'::jsonb,
                    outcome JSONB,
                    success BOOLEAN,
                    execution_time_ms INT,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    multi_criteria_analysis JSONB,
                    risk_assessment JSONB,
                    human_escalation_triggered BOOLEAN DEFAULT FALSE,
                    escalation_reason TEXT
                )
            """)

            # Create decision audit trail table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_decision_audit_trail (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    entry_id VARCHAR(255) UNIQUE NOT NULL,
                    decision_id VARCHAR(255) NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    event_type VARCHAR(50) NOT NULL,
                    actor VARCHAR(100) NOT NULL,
                    action TEXT NOT NULL,
                    context JSONB DEFAULT '{}'::jsonb,
                    changes JSONB DEFAULT '{}'::jsonb,
                    FOREIGN KEY (decision_id) REFERENCES ai_decision_history(decision_id)
                )
            """)

            # Create decision outcome tracking table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_decision_outcomes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    decision_id VARCHAR(255) NOT NULL,
                    actual_outcome JSONB NOT NULL,
                    expected_outcome JSONB NOT NULL,
                    variance_analysis JSONB,
                    success_score FLOAT,
                    lessons_learned TEXT[],
                    improvement_suggestions TEXT[],
                    recorded_at TIMESTAMPTZ DEFAULT NOW(),
                    FOREIGN KEY (decision_id) REFERENCES ai_decision_history(decision_id)
                )
            """)

            # Create decision optimization table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_decision_optimizations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    decision_type VARCHAR(50) NOT NULL,
                    optimization_type VARCHAR(50),
                    parameter_adjustments JSONB NOT NULL,
                    performance_improvement FLOAT,
                    applied_at TIMESTAMPTZ DEFAULT NOW(),
                    applied_by VARCHAR(100),
                    validation_results JSONB
                )
            """)

            # Create decision metrics table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_decision_metrics (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tree_id UUID REFERENCES ai_decision_trees(id),
                    decision_type VARCHAR(50),
                    total_decisions INT DEFAULT 0,
                    successful_decisions INT DEFAULT 0,
                    failed_decisions INT DEFAULT 0,
                    avg_confidence FLOAT DEFAULT 0.0,
                    avg_execution_time_ms FLOAT DEFAULT 0.0,
                    total_value_generated FLOAT DEFAULT 0.0,
                    total_cost_saved FLOAT DEFAULT 0.0,
                    period_start TIMESTAMPTZ DEFAULT NOW(),
                    period_end TIMESTAMPTZ,
                    metadata JSONB DEFAULT '{}'::jsonb
                )
            """)

            # Create decision rules table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_decision_rules (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    rule_name VARCHAR(255) UNIQUE NOT NULL,
                    rule_type VARCHAR(50),
                    condition TEXT NOT NULL,
                    action TEXT NOT NULL,
                    priority INT DEFAULT 50,
                    is_active BOOLEAN DEFAULT true,
                    applies_to_types TEXT[] DEFAULT '{}',
                    success_count INT DEFAULT 0,
                    failure_count INT DEFAULT 0,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            conn.commit()
            logger.info("Decision tree tables initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing decision tables: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _load_decision_trees(self):
        """Load pre-configured decision trees"""
        # Revenue decision tree
        self.decision_trees['revenue'] = self._create_revenue_tree()

        # Customer service tree
        self.decision_trees['customer'] = self._create_customer_tree()

        # Operational tree
        self.decision_trees['operational'] = self._create_operational_tree()

        # Emergency response tree
        self.decision_trees['emergency'] = self._create_emergency_tree()

        # Technical decision tree
        self.decision_trees['technical'] = self._create_technical_tree()

    def _create_revenue_tree(self) -> Dict:
        """Create revenue optimization decision tree"""
        return {
            'root': DecisionNode(
                node_id='revenue_root',
                node_type=DecisionType.FINANCIAL,
                question='Is there a revenue opportunity to pursue?',
                options=[
                    DecisionOption(
                        option_id='new_lead',
                        action=ActionType.EXECUTE,
                        description='Pursue new lead',
                        confidence=0.0,
                        expected_outcome={'revenue_potential': 'high'},
                        risks=['resource_allocation'],
                        requirements=['lead_data'],
                        estimated_duration=timedelta(days=7),
                        cost_estimate=500,
                        success_probability=0.0,
                        impact_score=0.0
                    ),
                    DecisionOption(
                        option_id='upsell_existing',
                        action=ActionType.EXECUTE,
                        description='Upsell existing customer',
                        confidence=0.0,
                        expected_outcome={'revenue_potential': 'medium'},
                        risks=['customer_churn'],
                        requirements=['customer_data'],
                        estimated_duration=timedelta(days=3),
                        cost_estimate=200,
                        success_probability=0.0,
                        impact_score=0.0
                    )
                ],
                parent_id=None,
                children=['revenue_qualify', 'revenue_pricing'],
                evaluation_criteria={'revenue_potential': 0.6, 'cost_efficiency': 0.4},
                threshold=0.7,
                metadata={}
            ),
            'revenue_qualify': DecisionNode(
                node_id='revenue_qualify',
                node_type=DecisionType.FINANCIAL,
                question='Does the opportunity meet qualification criteria?',
                options=[],
                parent_id='revenue_root',
                children=['revenue_execute'],
                evaluation_criteria={'fit_score': 0.5, 'budget_available': 0.5},
                threshold=0.6,
                metadata={}
            )
        }

    def _create_customer_tree(self) -> Dict:
        """Create customer service decision tree"""
        return {
            'root': DecisionNode(
                node_id='customer_root',
                node_type=DecisionType.CUSTOMER,
                question='What type of customer interaction is needed?',
                options=[
                    DecisionOption(
                        option_id='support_ticket',
                        action=ActionType.DELEGATE,
                        description='Handle support request',
                        confidence=0.0,
                        expected_outcome={'satisfaction': 'improved'},
                        risks=['escalation'],
                        requirements=['ticket_data'],
                        estimated_duration=timedelta(hours=2),
                        cost_estimate=50,
                        success_probability=0.0,
                        impact_score=0.0
                    ),
                    DecisionOption(
                        option_id='proactive_outreach',
                        action=ActionType.EXECUTE,
                        description='Proactive customer engagement',
                        confidence=0.0,
                        expected_outcome={'retention': 'improved'},
                        risks=['unwanted_contact'],
                        requirements=['customer_history'],
                        estimated_duration=timedelta(days=1),
                        cost_estimate=100,
                        success_probability=0.0,
                        impact_score=0.0
                    )
                ],
                parent_id=None,
                children=['customer_priority', 'customer_action'],
                evaluation_criteria={'urgency': 0.7, 'impact': 0.3},
                threshold=0.5,
                metadata={}
            )
        }

    def _create_operational_tree(self) -> Dict:
        """Create operational decision tree"""
        return {
            'root': DecisionNode(
                node_id='ops_root',
                node_type=DecisionType.OPERATIONAL,
                question='What operational optimization is needed?',
                options=[
                    DecisionOption(
                        option_id='resource_allocation',
                        action=ActionType.EXECUTE,
                        description='Optimize resource allocation',
                        confidence=0.0,
                        expected_outcome={'efficiency': 'improved'},
                        risks=['disruption'],
                        requirements=['resource_data'],
                        estimated_duration=timedelta(hours=4),
                        cost_estimate=200,
                        success_probability=0.0,
                        impact_score=0.0
                    ),
                    DecisionOption(
                        option_id='process_automation',
                        action=ActionType.SEQUENTIAL,
                        description='Automate manual process',
                        confidence=0.0,
                        expected_outcome={'automation': 'increased'},
                        risks=['implementation_failure'],
                        requirements=['process_mapping'],
                        estimated_duration=timedelta(days=5),
                        cost_estimate=1000,
                        success_probability=0.0,
                        impact_score=0.0
                    )
                ],
                parent_id=None,
                children=['ops_evaluate', 'ops_implement'],
                evaluation_criteria={'roi': 0.5, 'feasibility': 0.5},
                threshold=0.6,
                metadata={}
            )
        }

    def _create_emergency_tree(self) -> Dict:
        """Create emergency response decision tree"""
        return {
            'root': DecisionNode(
                node_id='emergency_root',
                node_type=DecisionType.EMERGENCY,
                question='What is the severity of the incident?',
                options=[
                    DecisionOption(
                        option_id='critical_response',
                        action=ActionType.ESCALATE,
                        description='Critical incident response',
                        confidence=1.0,
                        expected_outcome={'resolution': 'immediate'},
                        risks=['data_loss'],
                        requirements=['incident_data'],
                        estimated_duration=timedelta(minutes=15),
                        cost_estimate=0,
                        success_probability=0.9,
                        impact_score=1.0
                    ),
                    DecisionOption(
                        option_id='standard_response',
                        action=ActionType.EXECUTE,
                        description='Standard incident response',
                        confidence=0.8,
                        expected_outcome={'resolution': 'scheduled'},
                        risks=['delayed_resolution'],
                        requirements=['incident_data'],
                        estimated_duration=timedelta(hours=2),
                        cost_estimate=0,
                        success_probability=0.95,
                        impact_score=0.5
                    )
                ],
                parent_id=None,
                children=['emergency_triage', 'emergency_resolve'],
                evaluation_criteria={'severity': 0.8, 'impact': 0.2},
                threshold=0.7,
                metadata={}
            )
        }

    def _create_technical_tree(self) -> Dict:
        """Create technical decision tree"""
        return {
            'root': DecisionNode(
                node_id='tech_root',
                node_type=DecisionType.TECHNICAL,
                question='What technical action is required?',
                options=[
                    DecisionOption(
                        option_id='performance_optimization',
                        action=ActionType.EXECUTE,
                        description='Optimize system performance',
                        confidence=0.0,
                        expected_outcome={'performance': 'improved'},
                        risks=['instability'],
                        requirements=['metrics_data'],
                        estimated_duration=timedelta(hours=3),
                        cost_estimate=100,
                        success_probability=0.0,
                        impact_score=0.0
                    ),
                    DecisionOption(
                        option_id='scaling_decision',
                        action=ActionType.CONDITIONAL,
                        description='Scale infrastructure',
                        confidence=0.0,
                        expected_outcome={'capacity': 'increased'},
                        risks=['over_provisioning'],
                        requirements=['usage_data'],
                        estimated_duration=timedelta(hours=1),
                        cost_estimate=500,
                        success_probability=0.0,
                        impact_score=0.0
                    )
                ],
                parent_id=None,
                children=['tech_assess', 'tech_implement'],
                evaluation_criteria={'necessity': 0.6, 'cost_benefit': 0.4},
                threshold=0.5,
                metadata={}
            )
        }

    def _register_decision_handlers(self):
        """Register handlers for different decision types"""
        self.decision_handlers = {
            DecisionType.STRATEGIC: self._handle_strategic_decision,
            DecisionType.OPERATIONAL: self._handle_operational_decision,
            DecisionType.TACTICAL: self._handle_tactical_decision,
            DecisionType.EMERGENCY: self._handle_emergency_decision,
            DecisionType.FINANCIAL: self._handle_financial_decision,
            DecisionType.CUSTOMER: self._handle_customer_decision,
            DecisionType.TECHNICAL: self._handle_technical_decision,
            DecisionType.LEARNING: self._handle_learning_decision
        }

    async def make_decision(self, context: DecisionContext,
                           decision_type: DecisionType) -> DecisionResult:
        """Make an autonomous decision based on context"""
        decision_id = f"decision_{uuid.uuid4().hex[:8]}"
        audit_trail = []

        try:
            # Create audit entry for decision initiation
            audit_trail.append(self._create_audit_entry(
                decision_id, 'created', 'system', 'Decision process initiated',
                {'decision_type': decision_type.value, 'context': asdict(context)}
            ))

            # Select appropriate decision tree
            tree = self._select_decision_tree(decision_type, context)

            # Traverse tree to find best option
            selected_option, alternatives = await self._traverse_tree(tree, context)

            # Enhanced: Multi-criteria decision analysis
            multi_criteria = self._perform_multi_criteria_analysis(
                selected_option, alternatives, context
            )

            audit_trail.append(self._create_audit_entry(
                decision_id, 'evaluated', 'system', 'Multi-criteria analysis completed',
                {'criteria_count': len(multi_criteria.criteria_scores)}
            ))

            # Enhanced: Comprehensive risk assessment
            risk_assessment = self._assess_decision_risk(
                selected_option, context, decision_type
            )

            audit_trail.append(self._create_audit_entry(
                decision_id, 'evaluated', 'system', 'Risk assessment completed',
                {'risk_score': risk_assessment.overall_risk_score}
            ))

            # Evaluate confidence
            confidence_level = self._evaluate_confidence(selected_option, context)

            # Enhanced: Check if human escalation is needed
            human_escalation, escalation_reason = self._check_human_escalation(
                selected_option, risk_assessment, confidence_level, context
            )

            if human_escalation:
                audit_trail.append(self._create_audit_entry(
                    decision_id, 'escalated', 'system', 'Human review required',
                    {'reason': escalation_reason}
                ))

            # Generate reasoning
            reasoning = self._generate_reasoning(selected_option, context, tree)

            # Create execution plan
            execution_plan = self._create_execution_plan(selected_option, context)

            # Define monitoring plan
            monitoring_plan = self._create_monitoring_plan(selected_option)

            # Set success criteria
            success_criteria = self._define_success_criteria(selected_option)

            # Create rollback plan if needed
            rollback_plan = None
            if selected_option.confidence < 0.7 or risk_assessment.overall_risk_score > 0.7:
                rollback_plan = self._create_rollback_plan(selected_option)

            result = DecisionResult(
                decision_id=decision_id,
                timestamp=datetime.now(),
                selected_option=selected_option,
                confidence_level=confidence_level,
                reasoning=reasoning,
                alternative_options=alternatives,
                execution_plan=execution_plan,
                monitoring_plan=monitoring_plan,
                success_criteria=success_criteria,
                rollback_plan=rollback_plan,
                multi_criteria_analysis=multi_criteria,
                risk_assessment=risk_assessment,
                audit_trail=audit_trail,
                human_escalation_triggered=human_escalation,
                escalation_reason=escalation_reason
            )

            # Store decision with enhanced data
            self._store_decision(result, tree, context)

            # Store audit trail
            self._store_audit_trail(audit_trail)

            # Update metrics
            self._update_metrics(decision_type, result)

            return result

        except Exception as e:
            logger.error(f"Error making decision: {e}")
            audit_trail.append(self._create_audit_entry(
                decision_id, 'error', 'system', f'Decision failed: {str(e)}', {}
            ))
            return self._create_fallback_decision(decision_id, context, str(e))

    def _select_decision_tree(self, decision_type: DecisionType,
                             context: DecisionContext) -> Dict:
        """Select appropriate decision tree"""
        if decision_type == DecisionType.FINANCIAL:
            return self.decision_trees['revenue']
        elif decision_type == DecisionType.CUSTOMER:
            return self.decision_trees['customer']
        elif decision_type == DecisionType.OPERATIONAL:
            return self.decision_trees['operational']
        elif decision_type == DecisionType.EMERGENCY:
            return self.decision_trees['emergency']
        elif decision_type == DecisionType.TECHNICAL:
            return self.decision_trees['technical']
        else:
            # Default to operational tree
            return self.decision_trees['operational']

    async def _traverse_tree(self, tree: Dict,
                            context: DecisionContext) -> Tuple[DecisionOption, List[DecisionOption]]:
        """Traverse decision tree to find best option"""
        current_node = tree['root']
        selected_option = None
        alternatives = []

        while current_node:
            # Evaluate all options at this node
            scored_options = []
            for option in current_node.options:
                score = await self._score_option(option, context, current_node)
                option.confidence = score
                option.success_probability = self._calculate_success_probability(option, context)
                option.impact_score = self._calculate_impact_score(option, context)
                scored_options.append((score, option))

            # Sort by score
            scored_options.sort(key=lambda x: x[0], reverse=True)

            if scored_options:
                selected_option = scored_options[0][1]
                alternatives = [opt for _, opt in scored_options[1:3]]  # Top 2 alternatives

            # Check if we should continue traversing
            if current_node.children and selected_option and selected_option.confidence >= current_node.threshold:
                # Move to next node based on selected option
                next_node_id = current_node.children[0] if current_node.children else None
                current_node = tree.get(next_node_id)
            else:
                break

        return selected_option, alternatives

    async def _score_option(self, option: DecisionOption, context: DecisionContext,
                           node: DecisionNode) -> float:
        """Score a decision option based on context"""
        score = 0.0

        # Base scoring factors
        factors = {
            'resource_efficiency': self._evaluate_resource_efficiency(option, context),
            'time_efficiency': self._evaluate_time_efficiency(option, context),
            'risk_assessment': self._evaluate_risk(option, context),
            'expected_value': self._evaluate_expected_value(option, context),
            'alignment': self._evaluate_objective_alignment(option, context)
        }

        # Apply node-specific evaluation criteria
        for criterion, weight in node.evaluation_criteria.items():
            if criterion in factors:
                score += factors[criterion] * weight
            else:
                # Custom evaluation
                custom_score = self._custom_evaluation(criterion, option, context)
                score += custom_score * weight

        # Normalize score
        return min(max(score, 0.0), 1.0)

    def _evaluate_resource_efficiency(self, option: DecisionOption,
                                     context: DecisionContext) -> float:
        """Evaluate resource efficiency of option"""
        available = context.available_resources.get('budget', float('inf'))
        if available == 0:
            return 0.0
        efficiency = 1.0 - (option.cost_estimate / max(available, 1.0))
        return max(0.0, efficiency)

    def _evaluate_time_efficiency(self, option: DecisionOption,
                                  context: DecisionContext) -> float:
        """Evaluate time efficiency of option"""
        if not context.time_constraints:
            return 0.8  # Default good efficiency if no constraints

        available_time = context.time_constraints.total_seconds()
        required_time = option.estimated_duration.total_seconds()

        if required_time > available_time:
            return 0.0

        efficiency = 1.0 - (required_time / available_time)
        return efficiency

    def _evaluate_risk(self, option: DecisionOption, context: DecisionContext) -> float:
        """Evaluate risk level of option"""
        base_risk = len(option.risks) * 0.1
        risk_score = 1.0 - min(base_risk, 1.0)

        # Adjust for risk tolerance
        if context.risk_tolerance > 0.5:
            risk_score *= 1.2
        else:
            risk_score *= 0.8

        return min(max(risk_score, 0.0), 1.0)

    def _evaluate_expected_value(self, option: DecisionOption,
                                 context: DecisionContext) -> float:
        """Evaluate expected value of option"""
        # Simplified expected value calculation
        potential_value = option.expected_outcome.get('value', 1000)
        ev = (potential_value * option.success_probability) - option.cost_estimate

        # Normalize to 0-1 scale
        max_value = 10000  # Assumed max value
        normalized = (ev + max_value) / (2 * max_value)
        return min(max(normalized, 0.0), 1.0)

    def _evaluate_objective_alignment(self, option: DecisionOption,
                                     context: DecisionContext) -> float:
        """Evaluate how well option aligns with objectives"""
        if not context.objectives:
            return 0.5  # Neutral if no objectives

        alignment_score = 0.0
        for objective in context.objectives:
            # Check if option outcomes align with objective
            for outcome_key in option.expected_outcome:
                if objective.lower() in outcome_key.lower():
                    alignment_score += 1.0

        return min(alignment_score / len(context.objectives), 1.0)

    def _custom_evaluation(self, criterion: str, option: DecisionOption,
                          context: DecisionContext) -> float:
        """Custom evaluation for specific criteria"""
        # Add custom evaluation logic here
        return 0.5  # Default neutral score

    def _calculate_success_probability(self, option: DecisionOption,
                                      context: DecisionContext) -> float:
        """Calculate probability of success for option"""
        base_probability = 0.7  # Base success rate

        # Adjust based on requirements met
        requirements_met = sum(1 for req in option.requirements
                             if req in context.current_state)
        if option.requirements:
            requirement_factor = requirements_met / len(option.requirements)
            base_probability *= requirement_factor

        # Adjust based on historical success
        historical_factor = self._get_historical_success_rate(option.action)
        base_probability = (base_probability + historical_factor) / 2

        return min(max(base_probability, 0.0), 1.0)

    def _calculate_impact_score(self, option: DecisionOption,
                               context: DecisionContext) -> float:
        """Calculate potential impact of option"""
        impact_factors = {
            'revenue_impact': option.expected_outcome.get('revenue_potential', 0),
            'customer_impact': option.expected_outcome.get('satisfaction', 0),
            'efficiency_impact': option.expected_outcome.get('efficiency', 0)
        }

        # Weight impacts
        weighted_impact = sum(self._normalize_impact(v) for v in impact_factors.values())
        return min(weighted_impact / len(impact_factors), 1.0)

    def _normalize_impact(self, value: Any) -> float:
        """Normalize impact value to 0-1 scale"""
        if isinstance(value, str):
            impact_map = {'high': 1.0, 'medium': 0.5, 'low': 0.2, 'improved': 0.7}
            return impact_map.get(value.lower(), 0.5)
        elif isinstance(value, (int, float)):
            return min(abs(value) / 100, 1.0)
        return 0.5

    def _evaluate_confidence(self, option: DecisionOption,
                            context: DecisionContext) -> ConfidenceLevel:
        """Evaluate confidence level of decision"""
        if option.confidence > 0.8:
            return ConfidenceLevel.HIGH
        elif option.confidence > 0.5:
            return ConfidenceLevel.MEDIUM
        elif option.confidence > 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN

    def _generate_reasoning(self, option: DecisionOption, context: DecisionContext,
                          tree: Dict) -> str:
        """Generate reasoning for decision"""
        reasoning = f"Selected '{option.description}' based on: "

        factors = []
        if option.success_probability > 0.7:
            factors.append(f"high success probability ({option.success_probability:.1%})")
        if option.impact_score > 0.6:
            factors.append(f"significant expected impact ({option.impact_score:.1f})")
        if option.cost_estimate < context.available_resources.get('budget', float('inf')) * 0.2:
            factors.append("cost efficiency")
        if option.risks and len(option.risks) < 3:
            factors.append("acceptable risk level")

        reasoning += ", ".join(factors) if factors else "overall evaluation"
        reasoning += f". Confidence level: {option.confidence:.1%}"

        return reasoning

    def _create_execution_plan(self, option: DecisionOption,
                              context: DecisionContext) -> List[Dict]:
        """Create execution plan for selected option"""
        plan = []

        # Pre-execution steps
        plan.append({
            'step': 1,
            'action': 'validate_requirements',
            'description': 'Validate all requirements are met',
            'duration': timedelta(minutes=5),
            'dependencies': []
        })

        # Main execution
        if option.action == ActionType.SEQUENTIAL:
            # Break down into sequential steps
            sub_steps = self._create_sequential_steps(option)
            for i, step in enumerate(sub_steps, start=2):
                plan.append({
                    'step': i,
                    'action': step['action'],
                    'description': step['description'],
                    'duration': step['duration'],
                    'dependencies': [i-1] if i > 2 else [1]
                })
        elif option.action == ActionType.PARALLEL:
            # Create parallel execution branches
            branches = self._create_parallel_branches(option)
            plan.extend(branches)
        else:
            # Single execution step
            plan.append({
                'step': 2,
                'action': option.action.value,
                'description': option.description,
                'duration': option.estimated_duration,
                'dependencies': [1]
            })

        # Post-execution monitoring
        plan.append({
            'step': len(plan) + 1,
            'action': 'monitor_outcome',
            'description': 'Monitor execution outcome',
            'duration': timedelta(hours=1),
            'dependencies': [len(plan)]
        })

        return plan

    def _create_sequential_steps(self, option: DecisionOption) -> List[Dict]:
        """Create sequential execution steps"""
        # Simplified sequential step creation
        return [
            {'action': 'prepare', 'description': 'Prepare resources',
             'duration': timedelta(minutes=30)},
            {'action': 'execute', 'description': 'Execute main action',
             'duration': option.estimated_duration},
            {'action': 'verify', 'description': 'Verify results',
             'duration': timedelta(minutes=15)}
        ]

    def _create_parallel_branches(self, option: DecisionOption) -> List[Dict]:
        """Create parallel execution branches"""
        # Simplified parallel branch creation
        branches = []
        for i in range(2):  # Example: 2 parallel branches
            branches.append({
                'step': i + 2,
                'action': f'parallel_branch_{i+1}',
                'description': f'Execute parallel task {i+1}',
                'duration': option.estimated_duration,
                'dependencies': [1],
                'parallel': True
            })
        return branches

    def _create_monitoring_plan(self, option: DecisionOption) -> Dict:
        """Create monitoring plan for decision execution"""
        return {
            'monitoring_interval': timedelta(minutes=30),
            'metrics': [
                'execution_progress',
                'resource_utilization',
                'error_rate',
                'performance_indicators'
            ],
            'alert_conditions': [
                {'metric': 'error_rate', 'threshold': 0.1, 'action': 'escalate'},
                {'metric': 'execution_progress', 'threshold': 0.5, 'action': 'review'}
            ],
            'reporting_frequency': timedelta(hours=1)
        }

    def _define_success_criteria(self, option: DecisionOption) -> List[str]:
        """Define success criteria for decision"""
        criteria = []

        for outcome_key, outcome_value in option.expected_outcome.items():
            criteria.append(f"{outcome_key} achieved: {outcome_value}")

        criteria.append(f"Completed within {option.estimated_duration}")
        criteria.append(f"Cost not exceeding ${option.cost_estimate}")

        return criteria

    def _create_rollback_plan(self, option: DecisionOption) -> Dict:
        """Create rollback plan in case of failure"""
        return {
            'trigger_conditions': [
                'execution_failure',
                'unexpected_outcome',
                'resource_exhaustion'
            ],
            'rollback_steps': [
                {'action': 'stop_execution', 'duration': timedelta(minutes=5)},
                {'action': 'revert_changes', 'duration': timedelta(minutes=15)},
                {'action': 'notify_stakeholders', 'duration': timedelta(minutes=5)},
                {'action': 'document_lessons', 'duration': timedelta(minutes=10)}
            ],
            'recovery_strategy': 'return_to_previous_state'
        }

    def _create_fallback_decision(self, decision_id: str, context: DecisionContext,
                                 error: str) -> DecisionResult:
        """Create fallback decision in case of error"""
        fallback_option = DecisionOption(
            option_id='fallback',
            action=ActionType.ESCALATE,
            description='Escalate to human operator due to error',
            confidence=0.0,
            expected_outcome={'resolution': 'manual'},
            risks=['delayed_resolution'],
            requirements=[],
            estimated_duration=timedelta(hours=1),
            cost_estimate=0,
            success_probability=1.0,
            impact_score=0.5
        )

        return DecisionResult(
            decision_id=decision_id,
            timestamp=datetime.now(),
            selected_option=fallback_option,
            confidence_level=ConfidenceLevel.UNCERTAIN,
            reasoning=f"Fallback decision due to error: {error}",
            alternative_options=[],
            execution_plan=[{
                'step': 1,
                'action': 'escalate',
                'description': 'Escalate to human operator',
                'duration': timedelta(minutes=5),
                'dependencies': []
            }],
            monitoring_plan={'manual_monitoring': True},
            success_criteria=['Human resolution achieved'],
            rollback_plan=None
        )

    def _perform_multi_criteria_analysis(
        self,
        selected_option: DecisionOption,
        alternatives: List[DecisionOption],
        context: DecisionContext
    ) -> MultiCriteriaScore:
        """Perform multi-criteria decision analysis"""
        # Define decision criteria with weights
        criteria_weights = {
            'success_probability': 0.25,
            'impact_score': 0.20,
            'cost_efficiency': 0.20,
            'risk_level': 0.15,
            'time_efficiency': 0.10,
            'resource_efficiency': 0.10
        }

        # Score selected option on all criteria
        criteria_scores = {
            'success_probability': selected_option.success_probability,
            'impact_score': selected_option.impact_score,
            'cost_efficiency': 1.0 - (selected_option.cost_estimate / max(context.available_resources.get('budget', 1), 1)),
            'risk_level': 1.0 - (len(selected_option.risks) * 0.15),  # Inverse risk
            'time_efficiency': 1.0 - (selected_option.estimated_duration.total_seconds() / (context.time_constraints.total_seconds() if context.time_constraints else 86400)),
            'resource_efficiency': self._evaluate_resource_efficiency(selected_option, context)
        }

        # Normalize all scores to 0-1 range
        normalized_scores = {k: max(0.0, min(1.0, v)) for k, v in criteria_scores.items()}

        # Calculate weighted scores
        weighted_scores = {
            criterion: normalized_scores[criterion] * criteria_weights[criterion]
            for criterion in normalized_scores
        }

        total_score = sum(weighted_scores.values())

        # Sensitivity analysis: how much does each criterion affect the total?
        sensitivity_analysis = {}
        for criterion in criteria_weights:
            # Calculate score if this criterion was 0 vs 1
            impact = criteria_weights[criterion]
            sensitivity_analysis[criterion] = {
                'weight': criteria_weights[criterion],
                'current_contribution': weighted_scores[criterion],
                'max_potential_impact': impact,
                'sensitivity_ratio': weighted_scores[criterion] / impact if impact > 0 else 0
            }

        # Check if this is Pareto efficient (no alternative dominates in all criteria)
        pareto_efficient = True
        for alt in alternatives:
            alt_scores = {
                'success_probability': alt.success_probability,
                'impact_score': alt.impact_score,
                'cost_efficiency': 1.0 - (alt.cost_estimate / max(context.available_resources.get('budget', 1), 1)),
                'risk_level': 1.0 - (len(alt.risks) * 0.15),
                'time_efficiency': 1.0 - (alt.estimated_duration.total_seconds() / (context.time_constraints.total_seconds() if context.time_constraints else 86400)),
                'resource_efficiency': self._evaluate_resource_efficiency(alt, context)
            }
            # Check if alternative dominates in all criteria
            if all(alt_scores.get(k, 0) >= normalized_scores.get(k, 0) for k in normalized_scores):
                if any(alt_scores.get(k, 0) > normalized_scores.get(k, 0) for k in normalized_scores):
                    pareto_efficient = False
                    break

        return MultiCriteriaScore(
            criteria_scores=normalized_scores,
            weighted_scores=weighted_scores,
            total_score=total_score,
            sensitivity_analysis=sensitivity_analysis,
            pareto_efficient=pareto_efficient
        )

    def _assess_decision_risk(
        self,
        option: DecisionOption,
        context: DecisionContext,
        decision_type: DecisionType
    ) -> RiskAssessment:
        """Comprehensive risk assessment for a decision"""
        risk_categories = {}
        risk_factors = []
        mitigation_strategies = []

        # Financial risk
        financial_risk = min(option.cost_estimate / max(context.available_resources.get('budget', 1), 1), 1.0)
        risk_categories['financial'] = financial_risk
        if financial_risk > 0.7:
            risk_factors.append(f"High financial commitment: ${option.cost_estimate}")
            mitigation_strategies.append("Implement staged funding with milestones")

        # Execution risk (inverse of success probability)
        execution_risk = 1.0 - option.success_probability
        risk_categories['execution'] = execution_risk
        if execution_risk > 0.5:
            risk_factors.append(f"Low success probability: {option.success_probability:.1%}")
            mitigation_strategies.append("Develop detailed contingency plans")

        # Time risk
        if context.time_constraints:
            time_risk = option.estimated_duration.total_seconds() / context.time_constraints.total_seconds()
            risk_categories['time'] = min(time_risk, 1.0)
            if time_risk > 0.8:
                risk_factors.append("Tight timeline constraints")
                mitigation_strategies.append("Allocate buffer time and resources")

        # Impact risk (high impact = high risk if failure)
        impact_risk = option.impact_score * execution_risk
        risk_categories['impact'] = impact_risk
        if impact_risk > 0.6:
            risk_factors.append("High-impact decision with execution uncertainty")
            mitigation_strategies.append("Establish monitoring and early warning system")

        # Specific risk factors from option
        specific_risk = len(option.risks) * 0.15
        risk_categories['specific_factors'] = min(specific_risk, 1.0)
        if option.risks:
            risk_factors.extend(option.risks)
            mitigation_strategies.append("Address each identified risk factor individually")

        # Decision type specific risks
        if decision_type == DecisionType.FINANCIAL:
            risk_categories['regulatory'] = 0.3
            mitigation_strategies.append("Ensure compliance with financial regulations")
        elif decision_type == DecisionType.EMERGENCY:
            risk_categories['urgency'] = 0.8
            mitigation_strategies.append("Maintain rapid response capability")

        # Calculate overall risk score (weighted average)
        overall_risk = sum(risk_categories.values()) / len(risk_categories)

        # Determine escalation requirements
        escalation_required = overall_risk > 0.7
        human_review_required = overall_risk > 0.6 or len(risk_factors) > 5

        return RiskAssessment(
            overall_risk_score=overall_risk,
            risk_categories=risk_categories,
            mitigation_strategies=mitigation_strategies,
            escalation_required=escalation_required,
            human_review_required=human_review_required,
            risk_factors=risk_factors
        )

    def _check_human_escalation(
        self,
        option: DecisionOption,
        risk_assessment: RiskAssessment,
        confidence_level: ConfidenceLevel,
        context: DecisionContext
    ) -> Tuple[bool, Optional[str]]:
        """Determine if human escalation is needed"""
        escalation_reasons = []

        # High risk decisions
        if risk_assessment.overall_risk_score > 0.7:
            escalation_reasons.append(f"High risk score: {risk_assessment.overall_risk_score:.1%}")

        # Low confidence decisions
        if confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.UNCERTAIN]:
            escalation_reasons.append(f"Low confidence: {confidence_level.value}")

        # High-value decisions
        if option.cost_estimate > context.available_resources.get('budget', float('inf')) * 0.5:
            escalation_reasons.append(f"High cost: ${option.cost_estimate}")

        # Low success probability
        if option.success_probability < 0.5:
            escalation_reasons.append(f"Low success probability: {option.success_probability:.1%}")

        # Critical decision types
        if context.metadata.get('requires_approval', False):
            escalation_reasons.append("Explicit approval required")

        # Multiple high-severity risks
        if len([r for r in risk_assessment.risk_categories.values() if r > 0.7]) >= 2:
            escalation_reasons.append("Multiple high-severity risks detected")

        human_escalation = len(escalation_reasons) > 0
        escalation_reason = "; ".join(escalation_reasons) if escalation_reasons else None

        return human_escalation, escalation_reason

    def _create_audit_entry(
        self,
        decision_id: str,
        event_type: str,
        actor: str,
        action: str,
        context: Dict[str, Any],
        changes: Dict[str, Any] = None
    ) -> DecisionAuditEntry:
        """Create an audit trail entry"""
        return DecisionAuditEntry(
            entry_id=f"{decision_id}_{event_type}_{uuid.uuid4().hex[:8]}",
            decision_id=decision_id,
            timestamp=datetime.now(),
            event_type=event_type,
            actor=actor,
            action=action,
            context=context,
            changes=changes or {}
        )

    def _store_audit_trail(self, audit_trail: List[DecisionAuditEntry]):
        """Store audit trail entries in database"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            for entry in audit_trail:
                cur.execute("""
                    INSERT INTO ai_decision_audit_trail (
                        entry_id, decision_id, timestamp, event_type,
                        actor, action, context, changes
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    entry.entry_id,
                    entry.decision_id,
                    entry.timestamp,
                    entry.event_type,
                    entry.actor,
                    entry.action,
                    json.dumps(entry.context, default=str),
                    json.dumps(entry.changes, default=str)
                ))

            conn.commit()
        except Exception as e:
            logger.error(f"Error storing audit trail: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _store_decision(self, result: DecisionResult, tree: Dict,
                       context: DecisionContext):
        """Store decision in database with enhanced fields"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Serialize multi-criteria analysis and risk assessment
            multi_criteria_json = None
            if result.multi_criteria_analysis:
                multi_criteria_json = json.dumps({
                    'criteria_scores': result.multi_criteria_analysis.criteria_scores,
                    'weighted_scores': result.multi_criteria_analysis.weighted_scores,
                    'total_score': result.multi_criteria_analysis.total_score,
                    'sensitivity_analysis': result.multi_criteria_analysis.sensitivity_analysis,
                    'pareto_efficient': result.multi_criteria_analysis.pareto_efficient
                }, default=str)

            risk_assessment_json = None
            if result.risk_assessment:
                risk_assessment_json = json.dumps({
                    'overall_risk_score': result.risk_assessment.overall_risk_score,
                    'risk_categories': result.risk_assessment.risk_categories,
                    'mitigation_strategies': result.risk_assessment.mitigation_strategies,
                    'escalation_required': result.risk_assessment.escalation_required,
                    'human_review_required': result.risk_assessment.human_review_required,
                    'risk_factors': result.risk_assessment.risk_factors
                }, default=str)

            cur.execute("""
                INSERT INTO ai_decision_history (
                    decision_id, tree_id, node_path, context,
                    selected_option, confidence_level, reasoning,
                    execution_plan, timestamp, multi_criteria_analysis,
                    risk_assessment, human_escalation_triggered, escalation_reason
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                result.decision_id,
                None,  # Would need to look up tree_id
                ['root'],  # Simplified path
                json.dumps(asdict(context), default=str),
                json.dumps(asdict(result.selected_option), default=str),
                result.confidence_level.value,
                result.reasoning,
                json.dumps(result.execution_plan, default=str),
                result.timestamp,
                multi_criteria_json,
                risk_assessment_json,
                result.human_escalation_triggered,
                result.escalation_reason
            ))

            conn.commit()

        except Exception as e:
            logger.error(f"Error storing decision: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _update_metrics(self, decision_type: DecisionType, result: DecisionResult):
        """Update decision metrics"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Update or insert metrics
            cur.execute("""
                INSERT INTO ai_decision_metrics (
                    decision_type, total_decisions, avg_confidence
                ) VALUES (%s, 1, %s)
                ON CONFLICT (decision_type) DO UPDATE SET
                    total_decisions = ai_decision_metrics.total_decisions + 1,
                    avg_confidence = (
                        ai_decision_metrics.avg_confidence * ai_decision_metrics.total_decisions + %s
                    ) / (ai_decision_metrics.total_decisions + 1)
            """, (
                decision_type.value,
                result.selected_option.confidence,
                result.selected_option.confidence
            ))

            conn.commit()

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _get_historical_success_rate(self, action: ActionType) -> float:
        """Get historical success rate for action type from Real Database"""
        default_rates = {
            ActionType.EXECUTE: 0.75,
            ActionType.DELEGATE: 0.8,
            ActionType.MONITOR: 0.9,
            ActionType.ESCALATE: 0.95,
            ActionType.PARALLEL: 0.7,
            ActionType.SEQUENTIAL: 0.72,
            ActionType.CONDITIONAL: 0.68,
            ActionType.RETRY: 0.6
        }
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Query real historical performance
            cur.execute("""
                SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END)
                FROM ai_decision_history
                WHERE selected_option->>'action' = %s
                AND timestamp > NOW() - INTERVAL '30 days'
            """, (action.value,))
            
            result = cur.fetchone()
            real_rate = result[0] if result and result[0] is not None else None
            
            cur.close()
            conn.close()
            
            if real_rate is not None:
                return float(real_rate)
                
        except Exception as e:
            logger.warning(f"Failed to fetch historical rates for {action}: {e}")
            
        return default_rates.get(action, 0.7)

    # Handler methods for different decision types
    async def _handle_strategic_decision(self, context: DecisionContext) -> DecisionResult:
        """Handle strategic decisions"""
        return await self.make_decision(context, DecisionType.STRATEGIC)

    async def _handle_operational_decision(self, context: DecisionContext) -> DecisionResult:
        """Handle operational decisions"""
        return await self.make_decision(context, DecisionType.OPERATIONAL)

    async def _handle_tactical_decision(self, context: DecisionContext) -> DecisionResult:
        """Handle tactical decisions"""
        return await self.make_decision(context, DecisionType.TACTICAL)

    async def _handle_emergency_decision(self, context: DecisionContext) -> DecisionResult:
        """Handle emergency decisions"""
        return await self.make_decision(context, DecisionType.EMERGENCY)

    async def _handle_financial_decision(self, context: DecisionContext) -> DecisionResult:
        """Handle financial decisions"""
        return await self.make_decision(context, DecisionType.FINANCIAL)

    async def _handle_customer_decision(self, context: DecisionContext) -> DecisionResult:
        """Handle customer decisions"""
        return await self.make_decision(context, DecisionType.CUSTOMER)

    async def _handle_technical_decision(self, context: DecisionContext) -> DecisionResult:
        """Handle technical decisions"""
        return await self.make_decision(context, DecisionType.TECHNICAL)

    async def _handle_learning_decision(self, context: DecisionContext) -> DecisionResult:
        """Handle learning decisions"""
        return await self.make_decision(context, DecisionType.LEARNING)

    def record_decision_outcome(
        self,
        decision_id: str,
        actual_outcome: Dict[str, Any],
        success_score: float = None
    ):
        """Record the actual outcome of a decision for learning and optimization"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Fetch the original decision
            cur.execute("""
                SELECT selected_option, context FROM ai_decision_history
                WHERE decision_id = %s
            """, (decision_id,))

            row = cur.fetchone()
            if not row:
                logger.warning(f"Decision {decision_id} not found")
                return

            selected_option_data = json.loads(row[0])
            context_data = json.loads(row[1])
            expected_outcome = selected_option_data.get('expected_outcome', {})

            # Perform variance analysis
            variance_analysis = self._analyze_outcome_variance(
                expected_outcome, actual_outcome
            )

            # Extract lessons learned
            lessons_learned = self._extract_lessons_learned(
                variance_analysis, success_score or 0.5
            )

            # Generate improvement suggestions
            improvement_suggestions = self._generate_improvement_suggestions(
                variance_analysis, lessons_learned
            )

            # Calculate success score if not provided
            if success_score is None:
                success_score = self._calculate_success_score(
                    expected_outcome, actual_outcome
                )

            # Store outcome
            cur.execute("""
                INSERT INTO ai_decision_outcomes (
                    decision_id, actual_outcome, expected_outcome,
                    variance_analysis, success_score, lessons_learned,
                    improvement_suggestions, recorded_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                decision_id,
                json.dumps(actual_outcome, default=str),
                json.dumps(expected_outcome, default=str),
                json.dumps(variance_analysis, default=str),
                success_score,
                lessons_learned,
                improvement_suggestions
            ))

            # Update decision history with outcome
            cur.execute("""
                UPDATE ai_decision_history
                SET outcome = %s, success = %s
                WHERE decision_id = %s
            """, (json.dumps(actual_outcome, default=str), success_score > 0.6, decision_id))

            conn.commit()

            # Create audit entry
            audit_entry = self._create_audit_entry(
                decision_id, 'outcome_recorded', 'system',
                'Decision outcome recorded and analyzed',
                {'success_score': success_score, 'lessons_count': len(lessons_learned)}
            )
            self._store_audit_trail([audit_entry])

            # Trigger optimization if needed
            if success_score < 0.5:
                self._trigger_decision_optimization(decision_id, variance_analysis)

            # Add to learning buffer
            self.learning_buffer.append({
                'decision_id': decision_id,
                'outcome': actual_outcome,
                'success': success_score > 0.6,
                'timestamp': datetime.now(),
                'variance_analysis': variance_analysis
            })

            # Process learning if buffer is full
            if len(self.learning_buffer) >= 10:
                self._process_learning_buffer()

        except Exception as e:
            logger.error(f"Error recording outcome: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _analyze_outcome_variance(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze variance between expected and actual outcomes"""
        variance = {
            'matched_expectations': [],
            'exceeded_expectations': [],
            'fell_short': [],
            'unexpected_outcomes': []
        }

        # Check each expected outcome
        for key, expected_value in expected.items():
            actual_value = actual.get(key)

            if actual_value is None:
                variance['fell_short'].append({
                    'metric': key,
                    'expected': expected_value,
                    'actual': None,
                    'reason': 'Outcome not achieved'
                })
            elif isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                variance_pct = ((actual_value - expected_value) / max(expected_value, 1)) * 100
                if abs(variance_pct) < 10:
                    variance['matched_expectations'].append({
                        'metric': key,
                        'variance_pct': variance_pct
                    })
                elif variance_pct > 0:
                    variance['exceeded_expectations'].append({
                        'metric': key,
                        'expected': expected_value,
                        'actual': actual_value,
                        'variance_pct': variance_pct
                    })
                else:
                    variance['fell_short'].append({
                        'metric': key,
                        'expected': expected_value,
                        'actual': actual_value,
                        'variance_pct': variance_pct
                    })
            else:
                # String/categorical comparison
                if str(expected_value).lower() == str(actual_value).lower():
                    variance['matched_expectations'].append({'metric': key})
                else:
                    variance['fell_short'].append({
                        'metric': key,
                        'expected': expected_value,
                        'actual': actual_value
                    })

        # Check for unexpected outcomes
        for key in actual:
            if key not in expected:
                variance['unexpected_outcomes'].append({
                    'metric': key,
                    'value': actual[key]
                })

        return variance

    def _extract_lessons_learned(
        self,
        variance_analysis: Dict[str, Any],
        success_score: float
    ) -> List[str]:
        """Extract lessons learned from outcome variance"""
        lessons = []

        if success_score > 0.8:
            lessons.append("Decision process was highly effective")
            if variance_analysis.get('exceeded_expectations'):
                lessons.append("Outcomes exceeded expectations - consider adjusting future estimates upward")

        if success_score < 0.4:
            lessons.append("Decision process needs significant improvement")
            if len(variance_analysis.get('fell_short', [])) > 2:
                lessons.append("Multiple outcomes fell short - review decision criteria and risk assessment")

        if variance_analysis.get('unexpected_outcomes'):
            lessons.append("Decision analysis missed key factors - expand context gathering")

        fell_short = variance_analysis.get('fell_short', [])
        if any('cost' in item.get('metric', '').lower() for item in fell_short):
            lessons.append("Cost estimation accuracy needs improvement")

        if any('time' in item.get('metric', '').lower() for item in fell_short):
            lessons.append("Timeline estimation accuracy needs improvement")

        return lessons

    def _generate_improvement_suggestions(
        self,
        variance_analysis: Dict[str, Any],
        lessons_learned: List[str]
    ) -> List[str]:
        """Generate actionable improvement suggestions"""
        suggestions = []

        if len(variance_analysis.get('fell_short', [])) > 0:
            suggestions.append("Implement more conservative initial estimates")
            suggestions.append("Add additional validation checkpoints during execution")

        if variance_analysis.get('unexpected_outcomes'):
            suggestions.append("Expand pre-decision context gathering and analysis")
            suggestions.append("Include stakeholder feedback in decision process")

        if "cost" in str(lessons_learned).lower():
            suggestions.append("Develop more detailed cost breakdown models")
            suggestions.append("Add contingency buffers to financial estimates")

        if "time" in str(lessons_learned).lower():
            suggestions.append("Build timeline buffers for critical path items")
            suggestions.append("Track historical duration data for similar decisions")

        return suggestions

    def _calculate_success_score(
        self,
        expected: Dict[str, Any],
        actual: Dict[str, Any]
    ) -> float:
        """Calculate a 0-1 success score based on outcomes"""
        if not expected:
            return 0.5  # Neutral if no expectations

        total_score = 0.0
        metrics_count = 0

        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            metrics_count += 1

            if actual_value is None:
                continue  # 0 contribution for missing outcomes

            if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                # Numerical comparison - score based on how close actual is to expected
                if expected_value == 0:
                    score = 1.0 if actual_value == 0 else 0.5
                else:
                    ratio = actual_value / expected_value
                    score = 1.0 - abs(1.0 - ratio)  # Perfect = 1.0, further away = lower
                    score = max(0.0, min(1.0, score))
            else:
                # Categorical comparison
                score = 1.0 if str(actual_value).lower() == str(expected_value).lower() else 0.0

            total_score += score

        return total_score / metrics_count if metrics_count > 0 else 0.5

    def _trigger_decision_optimization(
        self,
        decision_id: str,
        variance_analysis: Dict[str, Any]
    ):
        """Trigger automatic optimization based on poor outcomes"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Identify which parameters need adjustment
            parameter_adjustments = {}

            fell_short = variance_analysis.get('fell_short', [])
            for item in fell_short:
                metric = item.get('metric', '')
                variance_pct = item.get('variance_pct', 0)

                if 'cost' in metric.lower():
                    parameter_adjustments['cost_estimation_buffer'] = abs(variance_pct) / 100
                elif 'time' in metric.lower():
                    parameter_adjustments['time_estimation_buffer'] = abs(variance_pct) / 100
                elif 'risk' in metric.lower():
                    parameter_adjustments['risk_sensitivity'] = 0.1  # Increase risk sensitivity

            if parameter_adjustments:
                cur.execute("""
                    INSERT INTO ai_decision_optimizations (
                        decision_type, optimization_type, parameter_adjustments,
                        performance_improvement, applied_by
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (
                    'general',  # Could be more specific based on decision context
                    'variance_correction',
                    json.dumps(parameter_adjustments, default=str),
                    None,  # Will be calculated after application
                    'auto_optimizer'
                ))

                conn.commit()
                logger.info(f"Optimization triggered for decision {decision_id}: {parameter_adjustments}")

        except Exception as e:
            logger.error(f"Error triggering optimization: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def learn_from_outcome(self, decision_id: str, outcome: Dict, success: bool):
        """Legacy method - redirects to enhanced outcome recording"""
        success_score = 1.0 if success else 0.0
        self.record_decision_outcome(decision_id, outcome, success_score)

    def _process_learning_buffer(self):
        """Process accumulated learning data"""
        if not self.learning_buffer:
            return

        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            if not self._learning_tables_initialized:
                self._ensure_learning_tables(cur)
                self._learning_tables_initialized = True

            total = len(self.learning_buffer)
            success_count = sum(1 for item in self.learning_buffer if item.get('success'))
            failure_count = total - success_count
            success_rate = success_count / total if total else 0.0

            failure_patterns = self._summarize_failure_patterns(self.learning_buffer)

            cur.execute("""
                INSERT INTO ai_learning_records (
                    agent_name,
                    learning_type,
                    description,
                    context,
                    confidence,
                    applied,
                    impact_score,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, false, %s, NOW())
            """, (
                "ai_decision_tree",
                "decision_outcome",
                "Aggregated learning buffer processing",
                json.dumps({
                    'total_samples': total,
                    'success_count': success_count,
                    'failure_count': failure_count,
                    'success_rate': success_rate,
                    'failure_patterns': failure_patterns,
                    'examples': self.learning_buffer[:3]
                }, default=str),
                success_rate,
                1.0 - success_rate
            ))

            self._update_agent_performance(cur, success_rate, total, failure_patterns)

            conn.commit()

            if failure_patterns:
                logger.warning(f"Learning buffer detected failure patterns: {failure_patterns}")

        except Exception as e:
            logger.error(f"Error processing learning buffer: {e}")
            if conn:
                conn.rollback()
        finally:
            self.learning_buffer.clear()
            if conn:
                conn.close()

    def _ensure_learning_tables(self, cur):
        """Ensure learning-related tables exist before writes"""
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ai_learning_records (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                agent_name TEXT,
                learning_type TEXT NOT NULL,
                description TEXT NOT NULL,
                context JSONB NOT NULL,
                confidence NUMERIC,
                applied BOOLEAN,
                applied_at TIMESTAMPTZ,
                impact_score NUMERIC,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS ai_agent_performance (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                project TEXT,
                agent_name TEXT,
                success_rate DOUBLE PRECISION,
                avg_response_time_ms DOUBLE PRECISION,
                total_interactions INT,
                performance_data JSONB,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

    def _summarize_failure_patterns(self, buffer: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Summarize recurring failure reasons for prompt improvements"""
        failure_counts = defaultdict(int)

        for item in buffer:
            if item.get('success') is False:
                outcome = item.get('outcome') or {}
                if isinstance(outcome, dict):
                    for key in ['error', 'reason', 'failure_reason', 'message', 'details']:
                        if outcome.get(key):
                            failure_counts[str(outcome[key]).strip()] += 1
                            break
                    else:
                        failure_counts[str(outcome)] += 1
                else:
                    failure_counts[str(outcome)] += 1

        patterns = [
            {'reason': reason, 'count': count}
            for reason, count in failure_counts.items() if reason
        ]

        return sorted(patterns, key=lambda x: x['count'], reverse=True)[:5]

    def _update_agent_performance(
        self,
        cur,
        success_rate: float,
        sample_size: int,
        failure_patterns: List[Dict[str, Any]]
    ):
        """Update aggregate agent performance metrics"""
        performance_payload = json.dumps({
            'success_rate': success_rate,
            'sample_size': sample_size,
            'failure_patterns': failure_patterns,
            'updated_at': datetime.utcnow().isoformat()
        }, default=str)

        cur.execute("""
            UPDATE ai_agent_performance
            SET success_rate = %s,
                total_interactions = COALESCE(total_interactions, 0) + %s,
                performance_data = %s::jsonb,
                updated_at = NOW()
            WHERE agent_name = %s
        """, (
            success_rate,
            sample_size,
            performance_payload,
            "ai_decision_tree"
        ))

        if cur.rowcount == 0:
            cur.execute("""
                INSERT INTO ai_agent_performance (
                    agent_name,
                    success_rate,
                    total_interactions,
                    performance_data,
                    updated_at
                ) VALUES (%s, %s, %s, %s::jsonb, NOW())
            """, (
                "ai_decision_tree",
                success_rate,
                sample_size,
                performance_payload
            ))

    def get_decision_stats(self) -> Dict[str, Any]:
        """Get decision statistics"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT
                    COUNT(*) as total_decisions,
                    AVG(CASE WHEN success THEN 1 ELSE 0 END) as success_rate,
                    AVG(execution_time_ms) as avg_execution_time
                FROM ai_decision_history
                WHERE timestamp > NOW() - INTERVAL '7 days'
            """)

            stats = cur.fetchone()

            return {
                'total_decisions': stats['total_decisions'] or 0,
                'success_rate': float(stats['success_rate'] or 0),
                'avg_execution_time_ms': float(stats['avg_execution_time'] or 0),
                'active_trees': len(self.decision_trees),
                'learning_buffer_size': len(self.learning_buffer)
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
        finally:
            if conn:
                conn.close()

    async def get_execution_guidance(
        self,
        agent_name: str,
        task_type: str,
        task_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get execution guidance for an agent task.

        Args:
            agent_name: Name of the agent requesting guidance
            task_type: Type of task being executed
            task_data: Optional task data

        Returns:
            Dict with confidence score and execution recommendations
        """
        try:
            # Map task types to decision trees
            tree_mapping = {
                'revenue': ['revenue', 'sales', 'pricing', 'lead'],
                'customer': ['customer', 'support', 'inquiry', 'complaint'],
                'operational': ['workflow', 'process', 'scheduling', 'resource'],
                'technical': ['code', 'deploy', 'fix', 'debug', 'monitor'],
                'emergency': ['alert', 'error', 'critical', 'outage']
            }

            # Determine appropriate decision tree
            selected_tree = 'operational'  # default
            task_type_lower = task_type.lower() if task_type else ''

            for tree_name, keywords in tree_mapping.items():
                if any(kw in task_type_lower for kw in keywords):
                    selected_tree = tree_name
                    break

            # Get historical success rate for this task type
            historical_success = self._get_historical_success_rate(
                ActionType.EXECUTE if 'execute' in task_type_lower else ActionType.MONITOR
            )

            # Base confidence calculation
            base_confidence = 0.75
            if self.decision_trees.get(selected_tree):
                base_confidence += 0.1  # Boost if we have a tree for this

            # Adjust based on historical performance
            confidence = min(1.0, base_confidence * (0.5 + historical_success * 0.5))

            return {
                'confidence': round(confidence, 3),
                'recommended_tree': selected_tree,
                'agent_name': agent_name,
                'task_type': task_type,
                'historical_success_rate': historical_success,
                'guidance': {
                    'proceed': confidence > 0.5,
                    'caution_level': 'low' if confidence > 0.7 else 'medium' if confidence > 0.5 else 'high',
                    'recommendations': [
                        f"Use {selected_tree} decision tree for optimal results",
                        "Monitor execution for early warning signs" if confidence < 0.7 else "Standard monitoring sufficient"
                    ]
                }
            }

        except Exception as e:
            logger.error(f"Error getting execution guidance: {e}")
            return {
                'confidence': 0.5,
                'error': str(e),
                'guidance': {'proceed': True, 'caution_level': 'medium'}
            }

    async def record_outcome(
        self,
        agent_name: str,
        task_type: str,
        success: bool,
        confidence_used: float = 0.5
    ) -> None:
        """
        Record the outcome of an agent execution for learning.
        This is the async interface expected by unified_system_integration.py.

        Args:
            agent_name: Name of the agent that executed
            task_type: Type of task executed
            success: Whether the execution succeeded
            confidence_used: The confidence score used for the decision
        """
        try:
            # Generate a decision ID for tracking
            decision_id = f"auto_{agent_name}_{task_type}_{uuid.uuid4().hex[:8]}"

            # Build outcome data
            actual_outcome = {
                'agent': agent_name,
                'task_type': task_type,
                'success': success,
                'confidence_used': confidence_used,
                'timestamp': datetime.now().isoformat()
            }

            # Calculate success score (0-1)
            success_score = 1.0 if success else 0.0

            # Use the existing record_decision_outcome method
            self.record_decision_outcome(
                decision_id=decision_id,
                actual_outcome=actual_outcome,
                success_score=success_score
            )

            logger.info(f"Recorded outcome for {agent_name}/{task_type}: success={success}")

        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")

    async def get_recovery_actions(
        self,
        agent_name: str,
        error_message: str
    ) -> List[str]:
        """
        Get suggested recovery actions for an agent error.

        Args:
            agent_name: Name of the agent that failed
            error_message: The error message

        Returns:
            List of suggested recovery actions
        """
        try:
            recovery_actions = []
            error_lower = error_message.lower()

            # Common error patterns and their recoveries
            if 'timeout' in error_lower:
                recovery_actions.append("Retry with increased timeout")
                recovery_actions.append("Check service availability")
            elif 'connection' in error_lower or 'network' in error_lower:
                recovery_actions.append("Retry connection")
                recovery_actions.append("Check network connectivity")
                recovery_actions.append("Verify service endpoints")
            elif 'permission' in error_lower or 'unauthorized' in error_lower:
                recovery_actions.append("Verify credentials")
                recovery_actions.append("Check API key validity")
            elif 'not found' in error_lower or '404' in error_lower:
                recovery_actions.append("Verify resource exists")
                recovery_actions.append("Check resource path/ID")
            elif 'validation' in error_lower or 'invalid' in error_lower:
                recovery_actions.append("Review input data format")
                recovery_actions.append("Check required fields")
            elif 'database' in error_lower or 'sql' in error_lower:
                recovery_actions.append("Retry database operation")
                recovery_actions.append("Check database connection pool")
            else:
                # Generic recovery suggestions
                recovery_actions.append("Retry operation with exponential backoff")
                recovery_actions.append("Check system logs for details")

            # Add agent-specific suggestions
            agent_lower = agent_name.lower()
            if 'deploy' in agent_lower:
                recovery_actions.append("Check deployment service status")
                recovery_actions.append("Verify build configuration")
            elif 'customer' in agent_lower or 'invoice' in agent_lower:
                recovery_actions.append("Verify customer data integrity")
            elif 'monitor' in agent_lower:
                recovery_actions.append("Check monitoring endpoints")

            return recovery_actions[:5]  # Return top 5 suggestions

        except Exception as e:
            logger.error(f"Failed to get recovery actions: {e}")
            return ["Retry operation", "Check system logs"]

# Singleton instance
_ai_decision_tree = None

def get_ai_decision_tree():
    """Get or create AI decision tree instance"""
    global _ai_decision_tree
    if _ai_decision_tree is None:
        _ai_decision_tree = AIDecisionTree()
    return _ai_decision_tree
