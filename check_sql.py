
import json
import os
import re
import glob

def load_schema(schema_path):
    with open(schema_path, 'r') as f:
        data = json.load(f)
    
    schema = {}
    for table in data.get('tables', []):
        table_name = table['name']
        columns = {col['name'] for col in table['columns']}
        schema[table_name] = columns
    return schema

def find_sql_mismatches(schema, root_dir):
    # Regex to capture INSERT INTO table (col1, col2, ...)
    # This is a basic regex and might miss complex cases or multiline SQL not handled perfectly
    insert_pattern = re.compile(r'INSERT\s+INTO\s+([a-zA-Z0-9_." ]+)\s*\(([^)]+)\)', re.IGNORECASE | re.DOTALL)
    
    mismatches = []
    
    for filepath in glob.glob(os.path.join(root_dir, '**/*.py'), recursive=True):
        try:
            with open(filepath, 'r') as f:
                content = f.read()
        except Exception:
            continue
            
        for match in insert_pattern.finditer(content):
            table_ref = match.group(1).strip().replace('"', '')
            # Handle schema.table notation
            if '.' in table_ref:
                table_name = table_ref.split('.')[-1]
            else:
                table_name = table_ref
                
            columns_str = match.group(2)
            # clean up newlines and extra spaces
            columns = [c.strip() for c in columns_str.split(',')]
            
            if table_name not in schema:
                # Some tables might be temporary or not in schema json if it's outdated
                # But for this report we flag them
                # check if it's a variable or f-string
                if '{' not in table_name and '$' not in table_name:
                     mismatches.append(f"File: {filepath}\n  Table '{table_name}' not found in schema.")
                continue
                
            table_cols = schema[table_name]
            for col in columns:
                clean_col = col.split()[0] # Handle cases like "col::type" or "col AS alias" though INSERT usually just has col
                clean_col = clean_col.strip()
                if clean_col and clean_col not in table_cols and '{' not in clean_col:
                     mismatches.append(f"File: {filepath}\n  Column '{clean_col}' not found in table '{table_name}'.")

    return mismatches

if __name__ == "__main__":
    schema = load_schema('database_schema.json')
    mismatches = find_sql_mismatches(schema, '.')
    
    if mismatches:
        print("Found SQL Mismatches:")
        for m in mismatches:
            print(m)
    else:
        print("No obvious SQL mismatches found with basic regex.")
