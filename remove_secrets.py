#!/usr/bin/env python3
"""
Remove all hardcoded secrets from Python files
"""

import re
from pathlib import Path

def remove_secrets_from_file(filepath):
    """Remove hardcoded secrets from a Python file"""

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content

    # Patterns to replace
    replacements = [
        # Database password
        (r'"password":\s*"<DB_PASSWORD_REDACTED>"', '"password": os.getenv("DB_PASSWORD")'),
        (r"'password':\s*'<DB_PASSWORD_REDACTED>'", "'password': os.getenv('DB_PASSWORD')"),
        (r'password\s*=\s*["\']<DB_PASSWORD_REDACTED>"\']', 'password=os.getenv("DB_PASSWORD")'),

        # Database host
        (r'"host":\s*["\']aws-0-us-east-2\.pooler\.supabase\.com["\']', '"host": os.getenv("DB_HOST")'),
        (r"'host':\s*['\"]aws-0-us-east-2\.pooler\.supabase\.com['\"]", "'host': os.getenv('DB_HOST')"),

        # Database user
        (r'"user":\s*["\']postgres\.yomagoqdmxszqtdwuhab["\']', '"user": os.getenv("DB_USER")'),
        (r"'user':\s*['\"]postgres\.yomagoqdmxszqtdwuhab['\"]", "'user': os.getenv('DB_USER')"),

        # Default password in getenv
        (r'os\.getenv\(["\']DB_PASSWORD["\']\s*,\s*["\']<DB_PASSWORD_REDACTED>"\']\)', 'os.getenv("DB_PASSWORD")'),

        # SendGrid API key
        (r'SENDGRID_API_KEY\s*=\s*["\']SG\.[^"\']+["\']', 'SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    # Add import if os is used but not imported
    if 'os.getenv' in content and 'import os' not in content:
        # Add import after the first line (usually shebang or docstring)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                lines.insert(i, 'import os')
                content = '\n'.join(lines)
                break

    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Remove secrets from all Python files"""

    print("Removing hardcoded secrets from Python files...")

    # Get all Python files
    python_files = list(Path('/home/matt-woodworth/brainops-ai-agents').glob('*.py'))

    modified_files = []

    for filepath in python_files:
        if filepath.name == 'remove_secrets.py':
            continue

        if remove_secrets_from_file(filepath):
            modified_files.append(filepath.name)
            print(f"✓ Cleaned: {filepath.name}")

    print(f"\nModified {len(modified_files)} files")

    if modified_files:
        print("\nModified files:")
        for f in modified_files:
            print(f"  - {f}")

    return len(modified_files)

if __name__ == "__main__":
    count = main()
    print(f"\nCompleted. {count} files were cleaned.")