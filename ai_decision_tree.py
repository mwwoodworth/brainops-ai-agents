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
                    timestamp TIMESTAMPTZ DEFAULT NOW()
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

        try:
            # Select appropriate decision tree
            tree = self._select_decision_tree(decision_type, context)

            # Traverse tree to find best option
            selected_option, alternatives = await self._traverse_tree(tree, context)

            # Evaluate confidence
            confidence_level = self._evaluate_confidence(selected_option, context)

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
            if selected_option.confidence < 0.7:
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
                rollback_plan=rollback_plan
            )

            # Store decision
            self._store_decision(result, tree, context)

            # Update metrics
            self._update_metrics(decision_type, result)

            return result

        except Exception as e:
            logger.error(f"Error making decision: {e}")
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

    def _store_decision(self, result: DecisionResult, tree: Dict,
                       context: DecisionContext):
        """Store decision in database"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ai_decision_history (
                    decision_id, tree_id, node_path, context,
                    selected_option, confidence_level, reasoning,
                    execution_plan, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                result.decision_id,
                None,  # Would need to look up tree_id
                ['root'],  # Simplified path
                json.dumps(asdict(context), default=str),
                json.dumps(asdict(result.selected_option), default=str),
                result.confidence_level.value,
                result.reasoning,
                json.dumps(result.execution_plan, default=str),
                result.timestamp
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
        """Get historical success rate for action type"""
        # Simplified - would query database for real rates
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

    def learn_from_outcome(self, decision_id: str, outcome: Dict, success: bool):
        """Learn from decision outcome to improve future decisions"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Update decision history with outcome
            cur.execute("""
                UPDATE ai_decision_history
                SET outcome = %s, success = %s
                WHERE decision_id = %s
            """, (json.dumps(outcome), success, decision_id))

            conn.commit()

            # Add to learning buffer
            self.learning_buffer.append({
                'decision_id': decision_id,
                'outcome': outcome,
                'success': success,
                'timestamp': datetime.now()
            })

            # Process learning if buffer is full
            if len(self.learning_buffer) >= 10:
                self._process_learning_buffer()

        except Exception as e:
            logger.error(f"Error learning from outcome: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

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

# Singleton instance
_ai_decision_tree = None

def get_ai_decision_tree():
    """Get or create AI decision tree instance"""
    global _ai_decision_tree
    if _ai_decision_tree is None:
        _ai_decision_tree = AIDecisionTree()
    return _ai_decision_tree
