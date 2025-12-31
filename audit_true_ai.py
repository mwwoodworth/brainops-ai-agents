#!/usr/bin/env python3
"""
Comprehensive audit of all systems for TRUE AI capabilities
Identifies mock data, predefined responses, and fake AI implementations
"""

import os
import sys
import re
import json
import glob
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any
from datetime import datetime

# Database configuration - NO hardcoded credentials
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),  # Required - no default
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER'),  # Required - no default
    'password': os.getenv('DB_PASSWORD'),  # Required - no default
    'port': int(os.getenv('DB_PORT', '5432'))
}

class AISystemAuditor:
    def __init__(self):
        self.findings = {
            'mock_data': [],
            'predefined_responses': [],
            'random_generators': [],
            'true_ai': [],
            'fake_ai': [],
            'llm_integrations': [],
            'recommendations': []
        }
        self.ai_agents_path = '/home/matt-woodworth/brainops-ai-agents'
        self.erp_path = '/home/matt-woodworth/myroofgenius-app'

    def audit_python_file(self, filepath: str) -> Dict[str, Any]:
        """Audit a Python file for AI authenticity"""
        findings = {
            'file': filepath,
            'has_mock': False,
            'has_predefined': False,
            'has_random': False,
            'has_true_ai': False,
            'issues': [],
            'ai_features': []
        }

        try:
            with open(filepath, 'r') as f:
                content = f.read()

            # Check for mock data patterns
            mock_patterns = [
                r'mock_[a-z_]+\s*=',
                r'MOCK_[A-Z_]+\s*=',
                r'fake_[a-z_]+\s*=',
                r'dummy_[a-z_]+\s*=',
                r'test_data\s*=\s*[\[\{]',
                r'return\s+["\'].*test.*["\']',
                r'response\s*=\s*["\'].*Lorem ipsum',
                r'sample_[a-z_]+\s*='
            ]

            for pattern in mock_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    findings['has_mock'] = True
                    findings['issues'].append(f"Mock data: {matches[0][:50]}")
                    self.findings['mock_data'].append({
                        'file': filepath,
                        'pattern': pattern,
                        'match': matches[0][:100]
                    })

            # Check for predefined responses
            predefined_patterns = [
                r'return\s+f?["\']Fixed response',
                r'response\s*=\s*["\']Completed successfully["\']',
                r'result\s*=\s*\{["\']status["\']\s*:\s*["\']success["\']',
                r'return\s+\{["\']analysis_id["\']\s*:.*["\']status["\']\s*:\s*["\']completed["\']',
                r'["\']result["\']\s*:\s*f?["\'].*completed for:',
                r'return\s+["\'].*Working.*["\']'
            ]

            for pattern in predefined_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    findings['has_predefined'] = True
                    findings['issues'].append(f"Predefined: {matches[0][:50]}")
                    self.findings['predefined_responses'].append({
                        'file': filepath,
                        'pattern': pattern,
                        'match': matches[0][:100]
                    })

            # Check for random/fake generation
            random_patterns = [
                r'random\.choice\(',
                r'random\.randint\(',
                r'random\.random\(',
                r'uuid\.uuid4\(\).*#.*fake',
                r'np\.random\.',
                r'randrange\(',
                r'return\s+random\.'
            ]

            for pattern in random_patterns:
                if re.search(pattern, content):
                    findings['has_random'] = True
                    findings['issues'].append(f"Random generation: {pattern}")
                    self.findings['random_generators'].append({
                        'file': filepath,
                        'pattern': pattern
                    })

            # Check for TRUE AI patterns
            true_ai_patterns = [
                r'openai\.',
                r'anthropic\.',
                r'model\.generate',
                r'llm\.',
                r'embedding',
                r'vector',
                r'transformer',
                r'completion',
                r'ChatCompletion',
                r'claude\.',
                r'gpt',
                r'langchain',
                r'langraph'
            ]

            for pattern in true_ai_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    findings['has_true_ai'] = True
                    findings['ai_features'].append(pattern)
                    self.findings['true_ai'].append({
                        'file': filepath,
                        'feature': pattern
                    })

            # Check for fake AI implementations
            fake_ai_patterns = [
                r'def\s+analyze.*:\s*\n\s*return\s+["\']',
                r'def\s+predict.*:\s*\n\s*return\s+\{',
                r'def\s+generate.*:\s*\n\s*return\s+["\']Test',
                r'# TODO:.*implement.*AI',
                r'# FIXME:.*real.*implementation',
                r'pass\s+#.*implement.*later'
            ]

            for pattern in fake_ai_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    findings['issues'].append(f"Fake AI: {pattern[:30]}")
                    self.findings['fake_ai'].append({
                        'file': filepath,
                        'pattern': pattern
                    })

        except Exception as e:
            findings['issues'].append(f"Error reading file: {e}")

        return findings

    def audit_all_python_files(self):
        """Audit all Python files in the system"""
        print("\n1. AUDITING PYTHON FILES FOR TRUE AI")
        print("-" * 50)

        python_files = glob.glob(f"{self.ai_agents_path}/*.py")

        for filepath in python_files:
            filename = os.path.basename(filepath)

            # Skip test and audit files
            if 'test' in filename.lower() or 'audit' in filename.lower():
                continue

            findings = self.audit_python_file(filepath)

            if findings['has_mock'] or findings['has_predefined'] or findings['has_random']:
                status = "⚠️"
            elif findings['has_true_ai']:
                status = "✅"
            else:
                status = "❌"

            issues = len(findings['issues'])
            ai_features = len(findings['ai_features'])

            print(f"{status} {filename:30} Issues: {issues}, AI Features: {ai_features}")

    def audit_database_for_mock_data(self):
        """Check database for mock/test data patterns"""
        print("\n2. AUDITING DATABASE FOR MOCK DATA")
        print("-" * 50)

        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Check for obvious test data in customers
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM customers
                WHERE
                    email LIKE '%test%' OR
                    email LIKE '%example.com' OR
                    name LIKE '%Test%' OR
                    name LIKE '%Demo%' OR
                    name LIKE '%Sample%'
            """)
            test_customers = cursor.fetchone()['count']

            if test_customers > 0:
                print(f"⚠️  Test customers found: {test_customers}")
                self.findings['mock_data'].append({
                    'type': 'database',
                    'table': 'customers',
                    'count': test_customers
                })
            else:
                print(f"✅ No obvious test customers")

            # Check for test patterns in AI tables
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM ai_master_context
                WHERE
                    context_key LIKE '%test%' OR
                    context_key LIKE '%mock%' OR
                    context_key LIKE '%demo%'
            """)
            test_context = cursor.fetchone()['count']

            if test_context > 0:
                print(f"⚠️  Test AI context entries: {test_context}")
            else:
                print(f"✅ No test AI context entries")

            # Check agent executions for patterns
            cursor.execute("""
                SELECT
                    agent_type,
                    COUNT(*) as count,
                    AVG(LENGTH(response)) as avg_response_length
                FROM agent_executions
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY agent_type
                ORDER BY count DESC
                LIMIT 10
            """)

            results = cursor.fetchall()
            print(f"\nRecent Agent Executions Analysis:")
            for row in results:
                avg_len = row['avg_response_length'] or 0
                if avg_len < 50:
                    status = "⚠️"  # Suspiciously short responses
                    self.findings['fake_ai'].append({
                        'agent': row['agent_type'],
                        'issue': f"Very short responses ({avg_len:.0f} chars avg)"
                    })
                else:
                    status = "✅"
                print(f"  {status} {row['agent_type']:20} Count: {row['count']:4}, Avg Response: {avg_len:.0f} chars")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Database audit error: {e}")

    def check_llm_integrations(self):
        """Check for actual LLM integrations"""
        print("\n3. CHECKING LLM INTEGRATIONS")
        print("-" * 50)

        # Check for API keys in environment
        api_keys = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'COHERE_API_KEY': os.getenv('COHERE_API_KEY'),
            'HUGGINGFACE_API_KEY': os.getenv('HUGGINGFACE_API_KEY')
        }

        for key_name, key_value in api_keys.items():
            if key_value:
                # Don't print the actual key
                print(f"✅ {key_name}: Configured ({len(key_value)} chars)")
                self.findings['llm_integrations'].append({
                    'type': 'api_key',
                    'name': key_name,
                    'configured': True
                })
            else:
                print(f"❌ {key_name}: Not configured")

        # Check for actual LLM usage in code
        print("\nLLM Usage in Code:")

        llm_files = []
        for pattern in ['*openai*.py', '*anthropic*.py', '*llm*.py', '*ai*.py']:
            llm_files.extend(glob.glob(f"{self.ai_agents_path}/{pattern}"))

        for filepath in set(llm_files):
            filename = os.path.basename(filepath)
            with open(filepath, 'r') as f:
                content = f.read()

            # Check for actual API calls
            if 'openai.ChatCompletion.create' in content or 'openai.Completion.create' in content:
                print(f"  ✅ {filename}: OpenAI API calls found")
                self.findings['llm_integrations'].append({
                    'file': filename,
                    'type': 'OpenAI',
                    'actual_calls': True
                })
            elif 'anthropic.Anthropic' in content or 'anthropic.Client' in content:
                print(f"  ✅ {filename}: Anthropic API calls found")
                self.findings['llm_integrations'].append({
                    'file': filename,
                    'type': 'Anthropic',
                    'actual_calls': True
                })
            elif 'model.generate' in content or 'llm(' in content:
                print(f"  ⚠️  {filename}: Generic LLM calls (needs verification)")
            else:
                if 'ai' in filename.lower() and filename not in ['audit_true_ai.py']:
                    print(f"  ❌ {filename}: No actual LLM calls found")
                    self.findings['fake_ai'].append({
                        'file': filename,
                        'issue': 'AI file without LLM calls'
                    })

    def analyze_ai_response_patterns(self):
        """Analyze response patterns to detect fake AI"""
        print("\n4. ANALYZING AI RESPONSE PATTERNS")
        print("-" * 50)

        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Check response diversity
            cursor.execute("""
                SELECT
                    agent_type,
                    COUNT(DISTINCT response) as unique_responses,
                    COUNT(*) as total_responses,
                    CAST(COUNT(DISTINCT response) AS FLOAT) / COUNT(*) * 100 as diversity_pct
                FROM agent_executions
                WHERE
                    created_at > NOW() - INTERVAL '7 days'
                    AND response IS NOT NULL
                GROUP BY agent_type
                HAVING COUNT(*) > 10
                ORDER BY diversity_pct DESC
            """)

            results = cursor.fetchall()
            print("Response Diversity Analysis:")

            for row in results:
                diversity = row['diversity_pct'] or 0
                if diversity < 20:
                    status = "❌"  # Very low diversity - likely predefined
                    self.findings['predefined_responses'].append({
                        'agent': row['agent_type'],
                        'diversity': f"{diversity:.1f}%",
                        'issue': 'Low response diversity'
                    })
                elif diversity < 50:
                    status = "⚠️"  # Medium diversity
                else:
                    status = "✅"  # High diversity - likely real AI

                print(f"  {status} {row['agent_type']:20} Diversity: {diversity:.1f}% ({row['unique_responses']}/{row['total_responses']})")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Response pattern analysis error: {e}")

    def check_specific_files(self):
        """Check specific critical files for true AI implementation"""
        print("\n5. CHECKING CRITICAL AI FILES")
        print("-" * 50)

        critical_files = [
            'app.py',
            'agent_executor.py',
            'ai_operating_system.py',
            'langgraph_orchestrator.py',
            'ai_decision_tree.py',
            'ai_pricing_engine.py',
            'customer_acquisition_agents.py'
        ]

        for filename in critical_files:
            filepath = os.path.join(self.ai_agents_path, filename)
            if not os.path.exists(filepath):
                print(f"  ❌ {filename}: File not found")
                continue

            with open(filepath, 'r') as f:
                content = f.read()

            issues = []
            ai_features = []

            # Check for specific problematic patterns
            if 'return {"status": "completed"' in content:
                issues.append("Hardcoded success response")
            if 'result = f"Analysis completed for:' in content:
                issues.append("Template response")
            if 'return {"analysis_id":' in content and 'uuid' in content:
                issues.append("Fake analysis response")

            # Check for real AI features
            if 'openai' in content.lower() or 'anthropic' in content.lower():
                ai_features.append("LLM integration")
            if 'embedding' in content.lower():
                ai_features.append("Embeddings")
            if 'vector' in content.lower():
                ai_features.append("Vector operations")
            if 'model.predict' in content or 'model.generate' in content:
                ai_features.append("Model inference")

            if issues and not ai_features:
                status = "❌"
            elif issues:
                status = "⚠️"
            else:
                status = "✅"

            print(f"  {status} {filename:30} Issues: {len(issues)}, AI Features: {len(ai_features)}")
            if issues:
                for issue in issues[:2]:
                    print(f"      - {issue}")

    def generate_recommendations(self):
        """Generate specific recommendations to make the system truly AI-powered"""
        print("\n6. RECOMMENDATIONS FOR TRUE AI POWER")
        print("-" * 50)

        recommendations = []

        # Based on findings, generate recommendations
        if len(self.findings['mock_data']) > 0:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Remove mock data generators',
                'files': len(set(f['file'] for f in self.findings['mock_data'])),
                'impact': 'Ensures real data processing'
            })

        if len(self.findings['predefined_responses']) > 0:
            recommendations.append({
                'priority': 'CRITICAL',
                'action': 'Replace predefined responses with LLM calls',
                'files': len(set(f['file'] for f in self.findings['predefined_responses'])),
                'impact': 'Enables dynamic AI responses'
            })

        if len(self.findings['fake_ai']) > 0:
            recommendations.append({
                'priority': 'CRITICAL',
                'action': 'Implement actual AI inference',
                'files': len(set(f.get('file', '') for f in self.findings['fake_ai'])),
                'impact': 'Provides genuine AI capabilities'
            })

        if not any('OPENAI' in i['name'] or 'ANTHROPIC' in i['name']
                   for i in self.findings['llm_integrations']
                   if i.get('configured')):
            recommendations.append({
                'priority': 'CRITICAL',
                'action': 'Configure LLM API keys',
                'files': 0,
                'impact': 'Enables AI model access'
            })

        for rec in recommendations:
            print(f"\n  [{rec['priority']}] {rec['action']}")
            if rec['files'] > 0:
                print(f"         Affects {rec['files']} files")
            print(f"         Impact: {rec['impact']}")

        self.findings['recommendations'] = recommendations

    def generate_report(self):
        """Generate comprehensive audit report"""
        print("\n" + "="*80)
        print("TRUE AI SYSTEM AUDIT REPORT")
        print("="*80)

        # Summary statistics
        total_issues = (
            len(self.findings['mock_data']) +
            len(self.findings['predefined_responses']) +
            len(self.findings['fake_ai'])
        )

        total_ai_features = len(self.findings['true_ai'])

        print(f"\nISSUES FOUND:")
        print(f"  Mock Data Instances:        {len(self.findings['mock_data'])}")
        print(f"  Predefined Responses:       {len(self.findings['predefined_responses'])}")
        print(f"  Fake AI Implementations:    {len(self.findings['fake_ai'])}")
        print(f"  Random Generators:          {len(self.findings['random_generators'])}")
        print(f"  TOTAL ISSUES:              {total_issues}")

        print(f"\nTRUE AI FEATURES:")
        print(f"  AI Feature Instances:       {total_ai_features}")
        print(f"  LLM Integrations:          {len(self.findings['llm_integrations'])}")

        # Calculate AI authenticity score
        if total_issues == 0 and total_ai_features > 20:
            score = 100
            verdict = "FULLY AI-POWERED"
        elif total_issues < 10 and total_ai_features > 10:
            score = 80
            verdict = "MOSTLY AI-POWERED"
        elif total_issues < 20 and total_ai_features > 5:
            score = 60
            verdict = "PARTIALLY AI-POWERED"
        elif total_ai_features > total_issues:
            score = 40
            verdict = "LIMITED AI CAPABILITIES"
        else:
            score = 20
            verdict = "MOSTLY FAKE AI"

        print(f"\n" + "="*50)
        print(f"AI AUTHENTICITY SCORE: {score}/100")
        print(f"VERDICT: {verdict}")
        print("="*50)

        if score < 80:
            print("\n⚠️  CRITICAL: System needs significant work to be truly AI-powered")
            print("\nTop Priority Actions:")
            for rec in self.findings['recommendations'][:3]:
                if rec['priority'] == 'CRITICAL':
                    print(f"  1. {rec['action']}")

        return score

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE TRUE AI SYSTEM AUDIT")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    auditor = AISystemAuditor()

    # Run all audits
    auditor.audit_all_python_files()
    auditor.audit_database_for_mock_data()
    auditor.check_llm_integrations()
    auditor.analyze_ai_response_patterns()
    auditor.check_specific_files()
    auditor.generate_recommendations()

    # Generate final report
    score = auditor.generate_report()

    # Save detailed findings
    findings_file = '/home/matt-woodworth/brainops-ai-agents/ai_audit_findings.json'
    with open(findings_file, 'w') as f:
        json.dump(auditor.findings, f, indent=2, default=str)

    print(f"\nDetailed findings saved to: {findings_file}")

    return 0 if score >= 80 else 1

if __name__ == "__main__":
    sys.exit(main())