import re
from typing import List, Tuple, Optional

class SimpleToken:
    def __init__(self, type_: str, value: str):
        self.type = type_
        self.value = value
    def __repr__(self):
        return f"({self.type}, {repr(self.value)})"

class SimpleSQLParser:
    """
    A lightweight, dependency-free SQL tokenizer/parser for security validation.
    Optimized for enforcing WHERE clause constraints on UPDATE/DELETE.
    """
    
    KEYWORDS = {"UPDATE", "DELETE", "INSERT", "INTO", "SET", "FROM", "WHERE", "AND", "OR", "WITH", "VALUES", "SELECT"}
    
    def __init__(self, sql: str):
        self.sql = sql
        self.tokens = self._tokenize(sql)
    
    def _tokenize(self, text: str) -> List[SimpleToken]:
        tokens = []
        i = 0
        n = len(text)
        
        while i < n:
            char = text[i]
            
            if char.isspace():
                i += 1
                continue
                
            # Comments
            if char == '-' and i+1 < n and text[i+1] == '-':
                end = text.find('\n', i)
                if end == -1: end = n
                i = end
                continue
            if char == '/' and i+1 < n and text[i+1] == '*':
                end = text.find('*/', i)
                if end == -1: raise ValueError("Unterminated comment")
                i = end + 2
                continue
                
            # Strings
            if char == "'":
                end = i + 1
                while end < n:
                    if text[end] == "'":
                        if end+1 < n and text[end+1] == "'":
                            end += 2 # Escaped quote
                            continue
                        else:
                            break
                    end += 1
                if end >= n: raise ValueError("Unterminated string")
                tokens.append(SimpleToken('STRING', text[i:end+1]))
                i = end + 1
                continue
                
            # Identifiers / Keywords
            if char.isalpha() or char == '_':
                end = i + 1
                while end < n and (text[end].isalnum() or text[end] == '_'):
                    end += 1
                word = text[i:end]
                if word.upper() in self.KEYWORDS:
                    tokens.append(SimpleToken('KEYWORD', word.upper()))
                else:
                    tokens.append(SimpleToken('IDENTIFIER', word))
                i = end
                continue
                
            # Symbols
            tokens.append(SimpleToken('SYMBOL', char))
            i += 1
            
        return tokens

    def validate_mutation_safety(self, required_column: str = "TENANT_ID"):
        """
        Ensures that if the query is a mutation (UPDATE/DELETE),
        it contains the required_column in the WHERE clause logic.
        """
        if not self.tokens: return
        
        first = self.tokens[0].value.upper()
        
        # 1. Block Common Table Expressions (WITH) completely for agents
        if first == "WITH":
            raise ValueError("CTE (WITH clause) not allowed in agent mutations.")
            
        if first not in ("UPDATE", "DELETE", "INSERT"):
            # Allow SELECTs (read-only handled by connection if possible, but here we focus on mutation safety)
            # If it's something else like TRUNCATE/DROP, we should block it.
            if first in ("TRUNCATE", "DROP", "ALTER"):
                raise ValueError(f"Dangerous operation {first} denied.")
            return

        # 2. INSERT Check
        if first == "INSERT":
            # Must contain tenant_id in the column list
            # Simplify: verify string presence in token stream for now (hard to parse INSERT INTO x (cols) VALUES ...)
            # A strict parser would map columns to values.
            # Fallback to token search for INSERT
            has_col = any(t.type == 'IDENTIFIER' and t.value.upper() == required_column.upper() for t in self.tokens)
            if not has_col:
                 raise ValueError(f"INSERT must include {required_column} column.")
            return

        # 3. UPDATE / DELETE Check
        # Must have a WHERE clause
        try:
            where_idx = next(i for i, t in enumerate(self.tokens) if t.type == 'KEYWORD' and t.value == 'WHERE')
        except StopIteration:
            raise ValueError(f"{first} requires a WHERE clause.")
            
        # Analyze WHERE clause tokens
        where_tokens = self.tokens[where_idx+1:]
        if not where_tokens:
            raise ValueError("Empty WHERE clause.")
            
        # Check for Tenant ID in WHERE
        # We enforce: Must see identifier 'tenant_id'
        has_tenant = any(t.type == 'IDENTIFIER' and t.value.upper() == required_column.upper() for t in where_tokens)
        if not has_tenant:
            raise ValueError(f"WHERE clause must constrain {required_column}.")
            
        # Check for OR (Draconian: no OR in WHERE allowed for agents)
        has_or = any(t.type == 'KEYWORD' and t.value == 'OR' for t in where_tokens)
        if has_or:
            raise ValueError("OR clauses not allowed in mutation WHERE block (safety restriction).")
