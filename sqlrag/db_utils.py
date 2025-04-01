import pandas as pd
import numpy as np
import json
import psycopg2
from psycopg2.extras import execute_values
import os
import time

from config import DB_PARAMS, SQL_TIMEOUT, MAX_RESULTS_ROWS

def get_db_connection():
    """Create and return database connection and cursor"""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    return conn, cur

def execute_sql_query(sql_query, timeout_seconds=SQL_TIMEOUT):
    """Execute SQL query with timeout and improved error handling"""
    conn = None
    cur = None
    
    try:
        # Get a fresh connection
        conn = psycopg2.connect(**DB_PARAMS)
        
        # Enable autocommit for session parameter changes
        conn.autocommit = True
        
        # Create cursor
        cur = conn.cursor()
        
        # Set statement timeout before starting transaction
        cur.execute(f"SET statement_timeout = {timeout_seconds * 1000};")  # milliseconds
        
        # Switch to transaction mode for the actual query
        conn.autocommit = False
        
        # Log the query being executed
        print(f"Executing SQL (with {timeout_seconds}s timeout): {sql_query[:200]}...")
        
        # Execute the query
        cur.execute(sql_query)
        
        # Get column names if the query returns results
        if cur.description:
            column_names = [desc[0] for desc in cur.description]
            
            # Fetch results with a row limit to avoid memory issues
            results = []
            while True:
                batch = cur.fetchmany(1000)  # Fetch in batches
                if not batch:
                    break
                results.extend(batch)
                
                # Check if we've fetched enough rows
                if len(results) >= MAX_RESULTS_ROWS:  # Set a reasonable maximum
                    print("Warning: Query returned more than 10,000 rows, truncating results")
                    break
            
            # Commit transaction
            conn.commit()
            
            # Convert results to DataFrame
            df = pd.DataFrame(results, columns=column_names)
            
            print(f"Query returned {len(df)} rows and {len(df.columns)} columns")
            
            return df
        else:
            # For queries that don't return results (e.g., INSERT, UPDATE)
            conn.commit()
            print("Query executed successfully (no results returned)")
            return pd.DataFrame()  # Empty DataFrame
    
    except psycopg2.Error as e:
        if conn:
            try:
                conn.rollback()
            except:
                pass
        
        error_msg = f"Error executing SQL query: {e}"
        print(error_msg)
        return None
    
    finally:
        # Clean up resources
        if cur:
            try:
                # Reset statement timeout if possible
                if conn and conn.status == psycopg2.extensions.STATUS_READY:
                    conn.autocommit = True
                    cur.execute("RESET statement_timeout;")
            except:
                pass
            cur.close()
        
        if conn:
            conn.close()

def validate_table_structure(table_name):
    """Get the actual column structure of a table, returns a list of column names"""
    conn, cur = get_db_connection()
    
    try:
        # Get column names for the table
        cur.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
        """)
        
        columns = [row[0] for row in cur.fetchall()]
        print(f"Columns in table {table_name}: {', '.join(columns)}")
        return columns
    
    except Exception as e:
        print(f"Error getting structure for table {table_name}: {e}")
        return []
    finally:
        cur.close()
        conn.close()

def check_query_columns(sql_query):
    """Analyze SQL query, validate that all referenced tables and columns exist"""
    import re
    
    # Extract tables used in the query
    from_pattern = re.compile(r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)', re.IGNORECASE)
    join_pattern = re.compile(r'\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)', re.IGNORECASE)
    
    tables = from_pattern.findall(sql_query) + join_pattern.findall(sql_query)
    tables = list(set(tables))  # Remove duplicates
    
    # Get the actual column structure for each table
    table_columns = {}
    for table in tables:
        table_columns[table] = validate_table_structure(table)
    
    # A simple method to find column references in the query
    # This is a simplified version; a complete SQL parser would be needed for full accuracy
    potential_issues = []
    
    for table in tables:
        columns = table_columns[table]
        # Look for patterns like "table.column"
        table_column_pattern = re.compile(rf'\b{table}\.([a-zA-Z_][a-zA-Z0-9_]*)', re.IGNORECASE)
        referenced_columns = table_column_pattern.findall(sql_query)
        
        for col in referenced_columns:
            if col not in columns:
                potential_issues.append(f"Warning: Column '{col}' does not exist in table '{table}'")
    
    return potential_issues, table_columns


def fetch_all_metadata():
    """Fetch metadata for all tables"""
    conn, cur = get_db_connection()
    
    try:
        # Get metadata for all tables
        cur.execute("""
            SELECT table_name, description, table_purpose, columns_info, 
                   primary_keys, foreign_keys, important_considerations,
                   common_joins, example_questions
            FROM mimic_table_metadata;
        """)
        
        all_metadata = cur.fetchall()
        
        # Format metadata as dictionary
        tables_metadata = {}
        for row in all_metadata:
            table_name = row[0]
            tables_metadata[table_name] = {
                'description': row[1],
                'table_purpose': row[2],
                'columns_info': row[3],
                'primary_keys': row[4],
                'foreign_keys': row[5],
                'important_considerations': row[6],
                'common_joins': row[7],
                'example_questions': row[8]
            }
        
        return tables_metadata
    
    finally:
        cur.close()
        conn.close()