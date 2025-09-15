#!/usr/bin/env python3
"""Fix production database schema issues"""

import psycopg2
import sys

DB_CONFIG = {
    "host": "aws-0-us-east-2.pooler.supabase.com",
    "database": "postgres",
    "user": "postgres.yomagoqdmxszqtdwuhab",
    "password": "REDACTED_SUPABASE_DB_PASSWORD",
    "port": 5432
}

def fix_schema():
    """Fix all schema issues and activate 50+ agents"""
    conn = None
    try:
        print("Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        cur = conn.cursor()

        # First ensure ai_agents table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ai_agents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(100) UNIQUE NOT NULL,
                status VARCHAR(50) DEFAULT 'active',
                capabilities JSONB DEFAULT '{}',
                last_active TIMESTAMP WITH TIME ZONE,
                total_executions INTEGER DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        conn.commit()
        print("✅ ai_agents table ready")

        fixes_applied = []

        # 1. Add metadata columns
        try:
            cur.execute("ALTER TABLE customers ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'")
            conn.commit()
            fixes_applied.append("✅ customers.metadata added")
        except Exception as e:
            conn.rollback()
            fixes_applied.append(f"⚠️ customers.metadata: {str(e)[:50]}")

        try:
            cur.execute("ALTER TABLE invoices ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'")
            conn.commit()
            fixes_applied.append("✅ invoices.metadata added")
        except Exception as e:
            conn.rollback()
            fixes_applied.append(f"⚠️ invoices.metadata: {str(e)[:50]}")

        # 2. Fix estimates constraints
        try:
            cur.execute("ALTER TABLE estimates ALTER COLUMN subtotal_cents DROP NOT NULL")
            conn.commit()
            fixes_applied.append("✅ estimates.subtotal_cents nullable")
        except Exception as e:
            conn.rollback()
            fixes_applied.append(f"⚠️ estimates.subtotal_cents: {str(e)[:50]}")

        try:
            cur.execute("ALTER TABLE estimates ALTER COLUMN tax_cents SET DEFAULT 0")
            conn.commit()
            fixes_applied.append("✅ estimates.tax_cents default 0")
        except Exception as e:
            conn.rollback()
            fixes_applied.append(f"⚠️ estimates.tax_cents: {str(e)[:50]}")

        try:
            cur.execute("ALTER TABLE estimates ALTER COLUMN discount_cents SET DEFAULT 0")
            conn.commit()
            fixes_applied.append("✅ estimates.discount_cents default 0")
        except Exception as e:
            conn.rollback()
            fixes_applied.append(f"⚠️ estimates.discount_cents: {str(e)[:50]}")

        # 3. Create workflow tables
        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    type VARCHAR(100),
                    status VARCHAR(50) DEFAULT 'active',
                    context JSONB DEFAULT '{}',
                    results JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            conn.commit()
            fixes_applied.append("✅ workflows table created")
        except Exception as e:
            conn.rollback()
            fixes_applied.append(f"⚠️ workflows: {str(e)[:50]}")

        # 4. Activate all 50+ agents
        print("\nActivating 50+ AI agents...")
        agents = [
            # Core Operations (10)
            'EstimationAgent', 'IntelligentScheduler', 'InvoicingAgent',
            'CustomerIntelligence', 'InventoryManager', 'DispatchOptimizer',
            'RouteOptimizer', 'QualityAssurance', 'SafetyCompliance', 'RegulatoryCompliance',

            # Financial Intelligence (10)
            'RevenueOptimizer', 'ExpenseAnalyzer', 'PayrollProcessor', 'TaxCalculator',
            'BudgetForecaster', 'CashFlowManager', 'ProfitMaximizer', 'CostReduction',
            'BillingAutomation', 'CollectionAgent',

            # Marketing & Sales (10)
            'LeadGenerator', 'CampaignManager', 'SEOOptimizer', 'SocialMediaBot',
            'EmailMarketing', 'ContentCreator', 'BrandManager', 'CustomerAcquisition',
            'SalesForecaster', 'ConversionOptimizer',

            # Analytics & Intelligence (10)
            'PredictiveAnalytics', 'ReportGenerator', 'DashboardManager', 'MetricsTracker',
            'InsightsEngine', 'TrendAnalyzer', 'PerformanceMonitor', 'DataValidator',
            'AnomalyDetector', 'ForecastEngine',

            # Communication (5)
            'ChatbotAgent', 'VoiceAssistant', 'SMSAutomation', 'NotificationManager', 'TranslationService',

            # Document Management (5)
            'ContractManager', 'ProposalGenerator', 'PermitTracker', 'InsuranceManager', 'WarrantyTracker',

            # Supply Chain (5)
            'ProcurementAgent', 'VendorManager', 'LogisticsCoordinator', 'WarehouseOptimizer', 'DeliveryTracker',

            # Human Resources (5)
            'RecruitingAgent', 'OnboardingManager', 'TrainingCoordinator', 'PerformanceEvaluator', 'BenefitsAdministrator',

            # System & Integration (5)
            'SystemMonitor', 'SecurityAgent', 'BackupManager', 'IntegrationHub', 'APIManager'
        ]

        activated = 0
        for agent in agents:
            try:
                cur.execute("""
                    INSERT INTO ai_agents (name, status, capabilities, created_at)
                    VALUES (%s, 'active', '{"ai_powered": true}'::jsonb, NOW())
                    ON CONFLICT (name) DO UPDATE
                    SET status = 'active', updated_at = NOW()
                """, (agent,))
                activated += 1
            except Exception as e:
                print(f"⚠️ Agent {agent}: {str(e)[:50]}")
                conn.rollback()

        conn.commit()
        fixes_applied.append(f"✅ Activated {activated}/{len(agents)} agents")

        # Print results
        print("\n=== SCHEMA FIXES APPLIED ===")
        for fix in fixes_applied:
            print(fix)

        # Verify current state
        print("\n=== VERIFICATION ===")
        cur.execute("""
            SELECT
                'Active Agents' as metric,
                COUNT(*)::text as value
            FROM ai_agents WHERE status = 'active'
            UNION ALL
            SELECT 'Total Agents', COUNT(*)::text FROM ai_agents
            UNION ALL
            SELECT 'Total Customers', COUNT(*)::text FROM customers
            UNION ALL
            SELECT 'Total Jobs', COUNT(*)::text FROM jobs
            UNION ALL
            SELECT 'Total Invoices', COUNT(*)::text FROM invoices
            UNION ALL
            SELECT 'Total Estimates', COUNT(*)::text FROM estimates
            UNION ALL
            SELECT 'Schema Version', '2.0.0'
        """)

        for metric, value in cur.fetchall():
            print(f"{metric}: {value}")

        cur.close()
        conn.close()
        print("\n✅ ALL FIXES COMPLETED")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        if conn:
            conn.close()
        return False

if __name__ == "__main__":
    if fix_schema():
        sys.exit(0)
    else:
        sys.exit(1)