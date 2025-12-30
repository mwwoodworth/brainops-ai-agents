-- ============================================================================
-- AI Tables Cleanup Migration
-- Generated: 2025-12-30
--
-- PURPOSE: Remove unused empty tables from the database
--
-- IMPORTANT: This migration is for REVIEW ONLY
-- DO NOT EXECUTE without explicit approval from the team
--
-- Before executing:
-- 1. Review TABLE_CLEANUP_REPORT.md
-- 2. Take a database backup
-- 3. Test in staging first
-- ============================================================================

-- Start transaction for safety
BEGIN;

-- ============================================================================
-- PHASE 1: Drop Unused Empty Tables (No Code References)
-- ============================================================================
-- These tables are completely empty and have NO references in the codebase

-- Acquisition/Lead tables (unused)
DROP TABLE IF EXISTS public.ai_acquisition_campaigns CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_acquisition_leads CASCADE;
-- Reason: Never populated, no code references

-- Activity tracking (unused)
DROP TABLE IF EXISTS public.ai_activities CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_agent_activity_tracking CASCADE;
-- Reason: Never populated, no code references

-- Agent management (unused)
DROP TABLE IF EXISTS public.ai_agent_collaborations CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_agent_configs CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_agent_connections CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_agent_metrics CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_agent_presence CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_agents_active CASCADE;
-- Reason: Never populated, use_devops_pipeline.py reference is just a list item

-- Alignment/Analysis (unused)
DROP TABLE IF EXISTS public.ai_alignment_checks CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_analysis_log CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_anomalies CASCADE;
-- Reason: Never populated, no code references

-- Assistant/Sessions (unused)
DROP TABLE IF EXISTS public.ai_assistance_sessions CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_assistant CASCADE;
-- Reason: Never populated, no code references

-- Automation (unused)
DROP TABLE IF EXISTS public.ai_automations_active CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_autonomous_config CASCADE;
-- Reason: Never populated, no code references

-- Board (unused partial)
DROP TABLE IF EXISTS public.ai_board_logs CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_board_sessions CASCADE;
-- Reason: Never populated, no code references

-- Campaigns (unused)
DROP TABLE IF EXISTS public.ai_campaign_templates CASCADE;
-- Reason: Never populated, no code references

-- Categorization (unused)
DROP TABLE IF EXISTS public.ai_categorization_feedback CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_categorization_suggestions CASCADE;
-- Reason: Never populated, no code references

-- Code quality (unused)
DROP TABLE IF EXISTS public.ai_code_changes CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_code_issues CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_code_metrics CASCADE;
-- Reason: Never populated, no code references

-- Collaboration (unused)
DROP TABLE IF EXISTS public.ai_collaboration_sessions CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_communication_log CASCADE;
-- Reason: Never populated, no code references

-- Competition (unused)
DROP TABLE IF EXISTS public.ai_competitive_positioning CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_competitor_intelligence CASCADE;
-- Reason: Never populated, no code references

-- Consensus (unused)
DROP TABLE IF EXISTS public.ai_consensus_decisions CASCADE;
-- Reason: Never populated, no code references

-- Context (unused)
DROP TABLE IF EXISTS public.ai_context CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_context_access_log CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_context_handoffs CASCADE;
-- Reason: Never populated, no code references

-- Conversations (unused partial)
DROP TABLE IF EXISTS public.ai_conversation_context CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_conversation_snapshots CASCADE;
-- Reason: Never populated, no code references

-- Cost (unused partial)
DROP TABLE IF EXISTS public.ai_cost_alerts CASCADE;
-- Reason: Never populated, no code references

-- CRM (unused)
DROP TABLE IF EXISTS public.ai_crm_action_approvals CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_crm_action_overrides CASCADE;
-- Reason: Never populated, no code references

-- Customer engagement (unused)
DROP TABLE IF EXISTS public.ai_customer_engagement CASCADE;
-- Reason: Never populated, no code references

-- Decision (unused partial)
DROP TABLE IF EXISTS public.ai_decision_engine CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_decision_log CASCADE;
-- Reason: Never populated, no code references (ai_decision_logs is different table)
DROP TABLE IF EXISTS public.ai_decision_points CASCADE;
-- Reason: Never populated, no code references

-- Deployment (unused)
DROP TABLE IF EXISTS public.ai_deployment_events CASCADE;
-- Reason: Never populated, no code references

-- Development (unused)
DROP TABLE IF EXISTS public.ai_development_sessions CASCADE;
-- Reason: Never populated, no code references

-- DevOps (unused)
DROP TABLE IF EXISTS public.ai_devops_insights CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_devops_optimizations CASCADE;
-- Reason: Never populated, no code references

-- Directives (unused)
DROP TABLE IF EXISTS public.ai_directives CASCADE;
-- Reason: Never populated, no code references

-- Dispatch (unused)
DROP TABLE IF EXISTS public.ai_dispatch_feedback CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_dispatch_recommendations CASCADE;
-- Reason: Never populated, no code references

-- Embeddings (unused - use document_embeddings or unified_ai_memory)
DROP TABLE IF EXISTS public.ai_embeddings CASCADE;
-- Reason: Never populated, legacy table

-- Estimates (unused)
DROP TABLE IF EXISTS public.ai_estimates CASCADE;
-- Reason: Never populated, no code references

-- Events (unused)
DROP TABLE IF EXISTS public.ai_events CASCADE;
-- Reason: Never populated, no code references

-- Feedback (unused - use ai_feedback_loop instead)
DROP TABLE IF EXISTS public.ai_feedback CASCADE;
-- Reason: Never populated, ai_feedback_loop is the active table

-- Generated (unused partial)
DROP TABLE IF EXISTS public.ai_generated_products CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_generations CASCADE;
-- Reason: Never populated, no code references

-- Improvement (unused)
DROP TABLE IF EXISTS public.ai_improvement_cycles CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_improvement_proposals CASCADE;
-- Reason: Never populated, no code references

-- Infrastructure (unused)
DROP TABLE IF EXISTS public.ai_infrastructure_alerts CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_infrastructure_metrics CASCADE;
-- Reason: Never populated, no code references

-- Knowledge (unused partial)
DROP TABLE IF EXISTS public.ai_knowledge_insights CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_knowledge_sync CASCADE;
-- Reason: Never populated, no code references

-- Learning (unused partial)
DROP TABLE IF EXISTS public.ai_learning_episodes CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_learning_events CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_learning_patterns CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_learning_sessions CASCADE;
-- Reason: Never populated, no code references

-- Logs (unused)
DROP TABLE IF EXISTS public.ai_logs CASCADE;
-- Reason: Never populated, no code references

-- Market (unused)
DROP TABLE IF EXISTS public.ai_market_insights CASCADE;
-- Reason: Never populated, no code references

-- Memory (LEGACY - use unified_ai_memory instead)
DROP TABLE IF EXISTS public.ai_memories CASCADE;
-- Reason: Legacy table, data should be in unified_ai_memory
DROP TABLE IF EXISTS public.ai_memory CASCADE;
-- Reason: Legacy table, data should be in unified_ai_memory
DROP TABLE IF EXISTS public.ai_memory_clusters CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_memory_enhanced CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_memory_feedback CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_memory_relationships CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_memory_store CASCADE;
-- Reason: Legacy table, data should be in unified_ai_memory
DROP TABLE IF EXISTS public.ai_memory_vectors CASCADE;
-- Reason: Legacy table, data should be in unified_ai_memory

-- Messages (unused)
DROP TABLE IF EXISTS public.ai_messages CASCADE;
-- Reason: Never populated, no code references

-- Models (unused)
DROP TABLE IF EXISTS public.ai_models CASCADE;
-- Reason: Never populated, no code references

-- Neural (unused)
DROP TABLE IF EXISTS public.ai_neural_pathways CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_neurons CASCADE;
-- Reason: Never populated, no code references

-- Next actions (unused)
DROP TABLE IF EXISTS public.ai_next_actions CASCADE;
-- Reason: Never populated, no code references

-- NLU (unused)
DROP TABLE IF EXISTS public.ai_nlu_interactions CASCADE;
-- Reason: Never populated, no code references

-- Nurture (unused partial)
DROP TABLE IF EXISTS public.ai_nurture_content_library CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_nurture_enrollments CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_nurture_executions CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_nurture_touchpoints CASCADE;
-- Reason: Never populated, no code references

-- Onboarding (unused partial)
DROP TABLE IF EXISTS public.ai_onboarding_analytics CASCADE;
-- Reason: Never populated, no code references

-- Operational (unused)
DROP TABLE IF EXISTS public.ai_operational_context CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_operations CASCADE;
-- Reason: Never populated, no code references

-- Optimizations (unused)
DROP TABLE IF EXISTS public.ai_optimizations CASCADE;
-- Reason: Never populated, no code references

-- Orchestration (unused partial)
DROP TABLE IF EXISTS public.ai_orchestration CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_orchestration_rules CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_orchestration_workflows CASCADE;
-- Reason: Never populated, no code references

-- OS metadata (unused)
DROP TABLE IF EXISTS public.ai_os_metadata CASCADE;
-- Reason: Never populated, no code references

-- Patterns (unused partial)
DROP TABLE IF EXISTS public.ai_pattern_recognition CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_patterns CASCADE;
-- Reason: Never populated, no code references

-- Persistence test (test artifact)
DROP TABLE IF EXISTS public.ai_persistence_test CASCADE;
-- Reason: Test table, should not be in production

-- Proposals (unused)
DROP TABLE IF EXISTS public.ai_proposals CASCADE;
-- Reason: Never populated, no code references

-- Providers (unused)
DROP TABLE IF EXISTS public.ai_providers CASCADE;
-- Reason: Never populated, no code references

-- Refactoring (unused)
DROP TABLE IF EXISTS public.ai_refactoring_opportunities CASCADE;
-- Reason: Never populated, no code references

-- Reports (unused partial)
DROP TABLE IF EXISTS public.ai_report_analytics CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_report_deliveries CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_report_templates CASCADE;
-- Reason: Never populated, no code references

-- Resource (unused)
DROP TABLE IF EXISTS public.ai_resource_usage CASCADE;
-- Reason: Never populated, no code references

-- Response cache (unused)
DROP TABLE IF EXISTS public.ai_response_cache CASCADE;
-- Reason: Never populated, no code references

-- Revenue (unused)
DROP TABLE IF EXISTS public.ai_revenue_leads CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_revenue_tracking CASCADE;
-- Reason: Never populated, no code references

-- Scheduling (unused)
DROP TABLE IF EXISTS public.ai_scheduling_predictions CASCADE;
-- Reason: Never populated, no code references

-- Sessions (unused)
DROP TABLE IF EXISTS public.ai_sessions CASCADE;
-- Reason: Never populated, no code references

-- Strategic (unused)
DROP TABLE IF EXISTS public.ai_strategic_goals CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_strategic_responses CASCADE;
-- Reason: Never populated, no code references

-- Synapses (unused)
DROP TABLE IF EXISTS public.ai_synapses CASCADE;
-- Reason: Never populated, no code references

-- System (unused partial)
DROP TABLE IF EXISTS public.ai_system_config CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_system_inefficiencies CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_system_metrics CASCADE;
-- Reason: Never populated, no code references

-- Task (unused partial)
DROP TABLE IF EXISTS public.ai_task_feedback CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_task_intelligence CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_task_manager CASCADE;
-- Reason: Never populated, no code references

-- Technical debt (unused)
DROP TABLE IF EXISTS public.ai_technical_debt CASCADE;
-- Reason: Never populated, no code references

-- ERP insights (unused)
DROP TABLE IF EXISTS public.ai_to_erp_insights CASCADE;
-- Reason: Never populated, no code references

-- Usage (unused)
DROP TABLE IF EXISTS public.ai_usage CASCADE;
-- Reason: Never populated, no code references

-- Vision (unused)
DROP TABLE IF EXISTS public.ai_vision_metrics CASCADE;
-- Reason: Never populated, no code references
DROP TABLE IF EXISTS public.ai_vision_progress CASCADE;
-- Reason: Never populated, no code references


-- ============================================================================
-- PHASE 2: Drop Duplicate Tables in Archive/Ops Schemas
-- ============================================================================
-- These are duplicates of tables that exist in the public schema

-- Archive schema duplicates (empty copies)
DROP TABLE IF EXISTS archive.ai_agent_executions CASCADE;
-- Reason: Duplicate of public.ai_agent_executions (which has 3606 rows)

DROP TABLE IF EXISTS archive.ai_autonomous_task_rules CASCADE;
-- Reason: Duplicate of public.ai_autonomous_task_rules (which has 5 rows)

DROP TABLE IF EXISTS archive.ai_onboarding_workflows CASCADE;
-- Reason: Duplicate of public.ai_onboarding_workflows (both empty)

-- Note: archive.ai_task_queue has 39 rows - REVIEW before dropping
-- Consider migrating data to public.ai_task_queue first
-- DROP TABLE IF EXISTS archive.ai_task_queue CASCADE;

DROP TABLE IF EXISTS archive.ai_tasks CASCADE;
-- Reason: Duplicate of public.ai_tasks (both empty)

-- Ops schema duplicates (empty)
DROP TABLE IF EXISTS ops.ai_workflows CASCADE;
-- Reason: Duplicate of public.ai_workflows (both empty)

-- Public empty duplicated tables
DROP TABLE IF EXISTS public.ai_workflows CASCADE;
-- Reason: Never populated, no code references (ai_workflow_executions is the active table)

DROP TABLE IF EXISTS public.ai_tasks CASCADE;
-- Reason: Never populated, no code references (ai_task_queue is the active table)


-- ============================================================================
-- PHASE 3: Review Tables (COMMENTED OUT - Require Manual Review)
-- ============================================================================
-- These tables have minimal data and may or may not be needed

-- DROP TABLE IF EXISTS public.ai_shared_context CASCADE;
-- Reason: 1 row, no code references - REVIEW if data is important

-- DROP TABLE IF EXISTS public.ai_orchestrations CASCADE;
-- Reason: 5 rows, no code references - REVIEW if data is important

-- DROP TABLE IF EXISTS public.ai_cash_flow_forecasts CASCADE;
-- Reason: 5 rows, no code references - REVIEW if data is important


-- ============================================================================
-- Commit Transaction
-- ============================================================================
-- Uncomment the line below to execute the changes
-- COMMIT;

-- For safety, rollback by default until reviewed
ROLLBACK;

-- ============================================================================
-- POST-CLEANUP VERIFICATION QUERIES
-- ============================================================================
-- Run these after the cleanup to verify results

-- Check remaining ai_* tables
-- SELECT relname as table_name, n_live_tup as row_count
-- FROM pg_stat_user_tables
-- WHERE relname LIKE 'ai_%'
-- ORDER BY n_live_tup DESC;

-- Count total ai_* tables
-- SELECT COUNT(*) as remaining_tables
-- FROM pg_stat_user_tables
-- WHERE relname LIKE 'ai_%';

-- ============================================================================
-- NOTES
-- ============================================================================
--
-- Tables NOT dropped (referenced in code, may be populated at runtime):
-- - ai_activity_feed
-- - ai_agent_performance
-- - ai_agent_registry
-- - ai_autonomous_actions
-- - ai_board_decisions
-- - ai_board_meetings
-- - ai_board_proposals
-- - ai_budgets
-- - ai_campaign_touches
-- - ai_churn_predictions
-- - ai_circuit_breakers
-- - ai_component_health
-- - ai_component_state
-- - ai_conversation_memory
-- - ai_conversations
-- - ai_cost_optimizations
-- - ai_customer_interactions
-- - ai_damage_assessments
-- - ai_decision_audit_trail
-- - ai_decision_history
-- - ai_decision_metrics
-- - ai_decision_nodes
-- - ai_decision_optimizations
-- - ai_decision_outcomes
-- - ai_decision_rules
-- - ai_decisions
-- - ai_detected_patterns
-- - ai_document_actions
-- - ai_document_analysis
-- - ai_document_content
-- - ai_document_embeddings
-- - ai_document_relationships
-- - ai_documents
-- - ai_email_deliveries
-- - ai_email_queue
-- - ai_emergency_events
-- - ai_error_logs
-- - ai_error_patterns
-- - ai_event_broadcasts
-- - ai_feedback_loop
-- - ai_followup_*
-- - ai_generated_reports
-- - ai_knowledge_base
-- - ai_knowledge_edges
-- - ai_knowledge_nodes
-- - ai_leads
-- - ai_learning_from_mistakes
-- - ai_learning_history
-- - ai_learning_records
-- - ai_master_context
-- - ai_model_performance
-- - ai_nurture_campaigns
-- - ai_nurture_engagement
-- - ai_nurture_metrics
-- - ai_nurture_sequences
-- - ai_onboarding_*
-- - ai_operations_log
-- - ai_outcome_patterns
-- - ai_permission_policies
-- - ai_personalizations
-- - ai_prediction_recommendations
-- - ai_predictions
-- - ai_proactive_*
-- - ai_realtime_events
-- - ai_realtime_subscriptions
-- - ai_reasoning_explanations
-- - ai_recovery_actions
-- - ai_recovery_actions_log
-- - ai_report_alerts
-- - ai_rollback_history
-- - ai_state_transitions
-- - ai_system_alerts
-- - ai_system_state
-- - ai_task_history
-- - ai_trace_spans
-- - ai_traces
-- - ai_trained_models
-- - ai_training_data
-- - ai_training_jobs
-- - ai_usage_logs
-- - ai_user_*
-- - ai_wake_triggers
-- - ai_workflow_executions
--
-- These tables are created by code and may be populated during runtime.
-- Monitor for 30 days before considering removal.
-- ============================================================================
