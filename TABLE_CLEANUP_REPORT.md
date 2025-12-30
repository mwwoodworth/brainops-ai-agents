# AI Tables Cleanup Report

**Generated:** 2025-12-30
**Total ai_* Tables:** 270
**Empty Tables (0 rows):** 235
**Low-Data Tables (1-5 rows):** 8

---

## Executive Summary

The database contains 270 tables with the `ai_*` prefix. Of these:
- **235 tables are completely empty** (0 rows)
- **8 tables have very few rows** (1-5 rows)
- **27 tables are actively used** with significant data

Many empty tables appear to be:
1. **Schema definitions created but never used** - Tables created in code with `CREATE TABLE IF NOT EXISTS` but no data inserted
2. **Legacy tables from old features** - Features that were planned but never implemented
3. **Duplicate tables in different schemas** - Same table name exists in `public`, `archive`, and `ops` schemas

---

## Duplicate Tables (Same Name, Different Schemas)

| Table Name | Schemas | Rows (by schema) | Recommendation |
|------------|---------|------------------|----------------|
| ai_agent_executions | public, archive | public: 3606, archive: 0 | DROP archive.ai_agent_executions |
| ai_autonomous_task_rules | public, archive | public: 5, archive: 0 | DROP archive.ai_autonomous_task_rules |
| ai_onboarding_workflows | public, archive | public: 0, archive: 0 | DROP both (unused) |
| ai_task_queue | public, archive | public: 7694, archive: 39 | MIGRATE archive data, DROP archive table |
| ai_tasks | public, archive | public: 0, archive: 0 | DROP both (unused) |
| ai_workflows | public, ops | public: 0, ops: 0 | DROP both (unused) |

---

## Tables with Active Data (DO NOT DROP)

These tables have significant data and are actively used:

| Table Name | Row Count | Status |
|------------|-----------|--------|
| ai_nerve_signals | 42,168 | KEEP - Active system |
| ai_thought_stream | 13,750 | KEEP - Active system |
| ai_vital_signs | 7,969 | KEEP - Active system |
| ai_task_queue | 7,694 | KEEP - Active system |
| ai_system_insights | 6,445 | KEEP - Active system |
| ai_customer_interventions | 5,380 | KEEP - Active system |
| ai_learning_insights | 4,657 | KEEP - Active system |
| ai_autonomous_tasks | 4,195 | KEEP - Active system |
| ai_consciousness_state | 4,040 | KEEP - Active system |
| ai_agent_executions | 3,606 | KEEP - Active system |
| ai_system_snapshot | 3,328 | KEEP - Active system |
| ai_development_tasks | 874 | KEEP - Active system |
| ai_business_insights | 539 | KEEP - Active system |
| ai_self_assessments | 429 | KEEP - Active system |
| ai_pricing_intelligence | 420 | KEEP - Active system |
| ai_scheduled_outreach | 315 | KEEP - Active system |
| ai_customer_health | 110 | KEEP - Active system |
| ai_attention_focus | 109 | KEEP - Active system |
| ai_agents | 65 | KEEP - Agent registry |
| ai_roofing_estimates | 61 | KEEP - Business data |
| ai_context_memory | 50 | KEEP - Memory system |
| ai_healing_rules | 39 | KEEP - Self-healing |
| ai_lead_scores | 17 | KEEP - Business data |
| ai_knowledge_graph | 16 | KEEP - Knowledge system |
| ai_insights | 10 | KEEP - Analytics |
| ai_operations_log | 8 | KEEP - Audit trail |

---

## Low-Data Tables (1-5 rows)

These tables exist but have minimal data:

| Table Name | Row Count | Referenced in Code | Recommendation |
|------------|-----------|-------------------|----------------|
| ai_shared_context | 1 | NO | REVIEW - May be deprecated |
| ai_schema_migrations | 1 | NO | KEEP - Migration tracking |
| ai_persistent_memory | 3 | YES (api/memory.py, app.py) | KEEP - Active memory table |
| ai_pricing_decisions | 4 | YES (ai_decision_tree.py) | KEEP - Pricing engine |
| ai_orchestrations | 5 | NO | REVIEW - May be deprecated |
| ai_cash_flow_forecasts | 5 | NO | REVIEW - May be unused |
| ai_autonomous_task_rules | 5 | YES (proactive_intelligence.py) | KEEP - Automation rules |
| ai_decision_trees | 5 | YES (ai_decision_tree.py) | KEEP - Decision engine |

---

## Empty Tables Analysis

### Category 1: SAFE TO DROP (No Code References)
These tables are empty and have NO references in the codebase:

| Table Name | Recommendation |
|------------|----------------|
| ai_acquisition_campaigns | DROP |
| ai_acquisition_leads | DROP |
| ai_activities | DROP |
| ai_agent_activity_tracking | DROP |
| ai_agent_collaborations | DROP |
| ai_agent_configs | DROP |
| ai_agent_connections | DROP |
| ai_agent_metrics | DROP |
| ai_agent_presence | DROP |
| ai_agents_active | DROP |
| ai_alignment_checks | DROP |
| ai_analysis_log | DROP |
| ai_anomalies | DROP |
| ai_assistance_sessions | DROP |
| ai_assistant | DROP |
| ai_automations_active | DROP |
| ai_autonomous_config | DROP |
| ai_board_logs | DROP |
| ai_board_sessions | DROP |
| ai_campaign_templates | DROP |
| ai_categorization_feedback | DROP |
| ai_categorization_suggestions | DROP |
| ai_code_changes | DROP |
| ai_code_issues | DROP |
| ai_code_metrics | DROP |
| ai_collaboration_sessions | DROP |
| ai_communication_log | DROP |
| ai_competitive_positioning | DROP |
| ai_competitor_intelligence | DROP |
| ai_consensus_decisions | DROP |
| ai_context | DROP |
| ai_context_access_log | DROP |
| ai_context_handoffs | DROP |
| ai_conversation_context | DROP |
| ai_conversation_snapshots | DROP |
| ai_cost_alerts | DROP |
| ai_crm_action_approvals | DROP |
| ai_crm_action_overrides | DROP |
| ai_customer_engagement | DROP |
| ai_decision_engine | DROP |
| ai_decision_points | DROP |
| ai_deployment_events | DROP |
| ai_development_sessions | DROP |
| ai_devops_insights | DROP |
| ai_devops_optimizations | DROP |
| ai_directives | DROP |
| ai_dispatch_feedback | DROP |
| ai_dispatch_recommendations | DROP |
| ai_embeddings | DROP |
| ai_estimates | DROP |
| ai_events | DROP |
| ai_feedback | DROP |
| ai_generated_products | DROP |
| ai_generations | DROP |
| ai_improvement_cycles | DROP |
| ai_improvement_proposals | DROP |
| ai_infrastructure_alerts | DROP |
| ai_infrastructure_metrics | DROP |
| ai_knowledge_insights | DROP |
| ai_knowledge_sync | DROP |
| ai_learning_episodes | DROP |
| ai_learning_events | DROP |
| ai_learning_patterns | DROP |
| ai_learning_sessions | DROP |
| ai_logs | DROP |
| ai_market_insights | DROP |
| ai_memories | DROP (legacy - use unified_ai_memory) |
| ai_memory | DROP (legacy - use unified_ai_memory) |
| ai_memory_clusters | DROP |
| ai_memory_enhanced | DROP |
| ai_memory_feedback | DROP |
| ai_memory_relationships | DROP |
| ai_memory_store | DROP (legacy - use unified_ai_memory) |
| ai_memory_vectors | DROP (legacy - use unified_ai_memory) |
| ai_messages | DROP |
| ai_models | DROP |
| ai_neural_pathways | DROP |
| ai_neurons | DROP |
| ai_next_actions | DROP |
| ai_nlu_interactions | DROP |
| ai_nurture_content_library | DROP |
| ai_nurture_enrollments | DROP |
| ai_nurture_executions | DROP |
| ai_nurture_touchpoints | DROP |
| ai_onboarding_analytics | DROP |
| ai_operational_context | DROP |
| ai_operations | DROP |
| ai_optimizations | DROP |
| ai_orchestration | DROP |
| ai_orchestration_rules | DROP |
| ai_orchestration_workflows | DROP |
| ai_os_metadata | DROP |
| ai_pattern_recognition | DROP |
| ai_patterns | DROP |
| ai_persistence_test | DROP |
| ai_proposals | DROP |
| ai_providers | DROP |
| ai_refactoring_opportunities | DROP |
| ai_report_analytics | DROP |
| ai_report_deliveries | DROP |
| ai_report_templates | DROP |
| ai_resource_usage | DROP |
| ai_response_cache | DROP |
| ai_revenue_leads | DROP |
| ai_revenue_tracking | DROP |
| ai_scheduling_predictions | DROP |
| ai_sessions | DROP |
| ai_strategic_goals | DROP |
| ai_strategic_responses | DROP |
| ai_synapses | DROP |
| ai_system_config | DROP |
| ai_system_inefficiencies | DROP |
| ai_system_metrics | DROP |
| ai_task_feedback | DROP |
| ai_task_intelligence | DROP |
| ai_task_manager | DROP |
| ai_technical_debt | DROP |
| ai_to_erp_insights | DROP |
| ai_usage | DROP |
| ai_vision_metrics | DROP |
| ai_vision_progress | DROP |

### Category 2: KEEP (Referenced in Code, Schema Only)
These tables are empty but have schema definitions and are referenced in code (tables may be populated at runtime):

| Table Name | Code Location | Recommendation |
|------------|---------------|----------------|
| ai_activity_feed | realtime_monitor.py | KEEP - Schema defined, used at runtime |
| ai_agent_performance | ai_decision_tree.py | KEEP - Performance tracking |
| ai_agent_registry | distributed_agent_coordination.py | KEEP - Agent coordination |
| ai_automation_rules | - | REVIEW |
| ai_autonomous_actions | proactive_intelligence.py | KEEP - Proactive actions |
| ai_board_decisions | board_action_pipeline.py, ai_board_governance.py | KEEP - Governance system |
| ai_board_meetings | ai_board_governance.py | KEEP - Governance system |
| ai_board_proposals | ai_board_governance.py | KEEP - Governance system |
| ai_budgets | ai_cost_optimization_engine.py | KEEP - Cost management |
| ai_campaign_touches | revenue_generation_system.py | KEEP - Revenue system |
| ai_churn_predictions | revenue_generation_system.py | KEEP - Churn prediction |
| ai_circuit_breakers | self_healing_error_recovery.py | KEEP - Error recovery |
| ai_component_health | self_healing_recovery.py | KEEP - Health monitoring |
| ai_component_state | system_state_manager.py | KEEP - State management |
| ai_conversation_memory | memory_system.py | KEEP - Memory system |
| ai_conversations | ai_knowledge_graph.py | KEEP - Knowledge graph |
| ai_cost_optimizations | ai_cost_optimization_engine.py | KEEP - Cost optimization |
| ai_customer_interactions | ai_training_pipeline.py | KEEP - Training data |
| ai_damage_assessments | industry_specific_ai_models.py | KEEP - Domain-specific |
| ai_decision_audit_trail | ai_decision_tree.py | KEEP - Audit trail |
| ai_decision_history | ai_decision_tree.py | KEEP - Decision tracking |
| ai_decision_logs | - | REVIEW |
| ai_decision_metrics | ai_decision_tree.py | KEEP - Metrics |
| ai_decision_nodes | ai_decision_tree.py | KEEP - Decision tree |
| ai_decision_optimizations | ai_decision_tree.py | KEEP - Optimization |
| ai_decision_outcomes | ai_decision_tree.py | KEEP - Outcome tracking |
| ai_decision_rules | ai_decision_tree.py | KEEP - Rules engine |
| ai_decisions | aurea_orchestrator.py, ai_decision_tree.py | KEEP - Central decision log |
| ai_detected_patterns | proactive_intelligence.py | KEEP - Pattern detection |
| ai_document_actions | document_processor.py | KEEP - Document processing |
| ai_document_analysis | document_processor.py | KEEP - Document processing |
| ai_document_content | document_processor.py | KEEP - Document processing |
| ai_document_embeddings | document_processor.py | KEEP - Document processing |
| ai_document_relationships | document_processor.py | KEEP - Document processing |
| ai_documents | document_processor.py | KEEP - Document processing |
| ai_email_deliveries | email_scheduler_daemon.py, email_sender.py | KEEP - Email system |
| ai_email_queue | lead_nurturing_system.py, email_sender.py | KEEP - Email system |
| ai_emergency_events | nerve_center.py | KEEP - Emergency handling |
| ai_error_logs | self_healing_recovery.py | KEEP - Error tracking |
| ai_error_patterns | self_healing_recovery.py | KEEP - Error patterns |
| ai_event_broadcasts | realtime_monitor.py | KEEP - Event system |
| ai_feedback_loop | ai_training_pipeline.py | KEEP - Training feedback |
| ai_followup_executions | intelligent_followup_system.py | KEEP - Follow-up system |
| ai_followup_metrics | intelligent_followup_system.py | KEEP - Follow-up metrics |
| ai_followup_responses | intelligent_followup_system.py | KEEP - Follow-up system |
| ai_followup_sequences | intelligent_followup_system.py | KEEP - Follow-up system |
| ai_followup_touchpoints | intelligent_followup_system.py | KEEP - Follow-up system |
| ai_generated_reports | automated_reporting_system.py | KEEP - Reporting system |
| ai_knowledge_base | memory_system.py | KEEP - Knowledge base |
| ai_knowledge_edges | ai_knowledge_graph.py | KEEP - Knowledge graph |
| ai_knowledge_nodes | ai_knowledge_graph.py | KEEP - Knowledge graph |
| ai_leads | lead_nurturing_system.py | KEEP - Lead management |
| ai_learning_from_mistakes | ai_self_awareness.py | KEEP - Self-improvement |
| ai_learning_history | ai_training_pipeline.py | KEEP - Learning history |
| ai_learning_records | ai_decision_tree.py | KEEP - Learning records |
| ai_master_context | memory_system.py, app.py | KEEP - Master context |
| ai_model_performance | multi_model_consensus.py | KEEP - Model metrics |
| ai_nurture_campaigns | revenue_generation_system.py | KEEP - Nurture campaigns |
| ai_nurture_engagement | lead_nurturing_system.py | KEEP - Engagement tracking |
| ai_nurture_metrics | lead_nurturing_system.py | KEEP - Nurture metrics |
| ai_nurture_sequences | lead_nurturing_system.py | KEEP - Nurture sequences |
| ai_onboarding_actions | ai_customer_onboarding.py | KEEP - Onboarding system |
| ai_onboarding_events | ai_customer_onboarding.py | KEEP - Onboarding system |
| ai_onboarding_interventions | ai_customer_onboarding.py | KEEP - Onboarding system |
| ai_onboarding_journeys | ai_customer_onboarding.py | KEEP - Onboarding system |
| ai_onboarding_stages | ai_customer_onboarding.py | KEEP - Onboarding system |
| ai_onboarding_steps | revenue_generation_system.py | KEEP - Onboarding system |
| ai_onboarding_tasks | ai_customer_onboarding.py | KEEP - Onboarding system |
| ai_onboarding_workflows | revenue_generation_system.py | KEEP - Onboarding system |
| ai_operations_log | ai_integration_layer.py | KEEP - Operations log |
| ai_outcome_patterns | ai_training_pipeline.py | KEEP - Pattern recognition |
| ai_permission_policies | ai_context_awareness.py | KEEP - Permissions |
| ai_personalizations | ai_context_awareness.py | KEEP - Personalization |
| ai_prediction_recommendations | predictive_analytics_engine.py | KEEP - Predictions |
| ai_predictions | predictive_analytics_engine.py | KEEP - Predictions |
| ai_proactive_health | self_healing_recovery.py | KEEP - Health monitoring |
| ai_proactive_learnings | proactive_intelligence.py | KEEP - Learning system |
| ai_proactive_predictions | proactive_intelligence.py | KEEP - Predictions |
| ai_realtime_events | realtime_monitor.py | KEEP - Realtime events |
| ai_realtime_subscriptions | realtime_monitor.py | KEEP - Subscriptions |
| ai_reasoning_explanations | ai_self_awareness.py | KEEP - Explainability |
| ai_recovery_actions | self_healing_error_recovery.py | KEEP - Recovery system |
| ai_recovery_actions_log | self_healing_recovery.py | KEEP - Recovery log |
| ai_report_alerts | automated_reporting_system.py | KEEP - Report alerts |
| ai_rollback_history | self_healing_recovery.py | KEEP - Rollback tracking |
| ai_state_transitions | system_state_manager.py | KEEP - State management |
| ai_system_alerts | system_state_manager.py | KEEP - Alerts |
| ai_system_state | system_state_manager.py, ai_integration_layer.py | KEEP - System state |
| ai_task_history | memory_system.py | KEEP - Task history |
| ai_tasks | - | REVIEW - May be unused |
| ai_trace_spans | ai_tracer.py | KEEP - Tracing |
| ai_traces | ai_tracer.py | KEEP - Tracing |
| ai_trained_models | ai_training_pipeline.py | KEEP - Model registry |
| ai_training_data | ai_training_pipeline.py | KEEP - Training data |
| ai_training_jobs | ai_training_pipeline.py | KEEP - Training jobs |
| ai_usage_logs | ai_core.py | KEEP - Usage tracking |
| ai_user_context | ai_context_awareness.py | KEEP - User context |
| ai_user_embeddings | ai_context_awareness.py | KEEP - User embeddings |
| ai_user_interactions | ai_context_awareness.py | KEEP - User interactions |
| ai_user_preferences | ai_context_awareness.py | KEEP - User preferences |
| ai_user_profiles | ai_context_awareness.py | KEEP - User profiles |
| ai_user_sessions | ai_context_awareness.py | KEEP - User sessions |
| ai_wake_triggers | alive_core.py | KEEP - Wake system |
| ai_workflow_executions | ai_workflow_templates.py | KEEP - Workflow system |
| ai_workflows | - | REVIEW - May be unused |

---

## Cleanup Statistics

| Category | Count | Action |
|----------|-------|--------|
| Active Tables (>5 rows) | 27 | DO NOT DROP |
| Low-Data Tables (1-5 rows) | 8 | REVIEW individually |
| Empty but Referenced | ~100 | KEEP (schema definitions) |
| Safe to Drop | ~110 | DROP in migration |
| Duplicate Tables | 12 (6 pairs) | CONSOLIDATE |

---

## Recommended Actions

1. **Phase 1: Drop Unused Empty Tables** (~110 tables)
   - Run `migrations/cleanup_empty_tables.sql`
   - These have no code references and no data

2. **Phase 2: Consolidate Duplicate Tables** (6 tables)
   - Migrate any data from archive/ops schemas to public
   - Drop duplicate schema tables

3. **Phase 3: Review Low-Data Tables** (8 tables)
   - Determine if data should be migrated or tables dropped

4. **Phase 4: Monitor Referenced Empty Tables**
   - These may be populated at runtime
   - Track if they remain empty after 30 days

---

## Next Steps

1. Review this report with the team
2. Execute `migrations/cleanup_empty_tables.sql` after approval
3. Monitor application logs for any table access errors
4. Update codebase to remove references to dropped tables

---

*Report generated automatically. Review all recommendations before executing any DROP statements.*
