import pandas as pd
import numpy as np
import json
import psycopg2
from psycopg2.extras import execute_values
from openai import OpenAI
import os
import time
from dotenv import load_dotenv
from config import OPENAI_API_KEY

# OpenAI API configuration
# Initialize the client
client = OpenAI(api_key=OPENAI_API_KEY)

def format_metadata_for_prompt(metadata):
    """Format metadata into text suitable for LLM prompt"""
    formatted_text = "# Database Schema Information\n\n"
    
    for table_name, table_info in metadata.items():
        formatted_text += f"## Table: {table_name}\n"
        formatted_text += f"Description: {table_info['description']}\n"
        formatted_text += f"Purpose: {table_info['table_purpose']}\n\n"
        
        # Add primary key information
        if table_info['primary_keys']:
            formatted_text += f"Primary Keys: {', '.join(table_info['primary_keys'])}\n\n"
        
        # Add foreign key information
        if table_info['foreign_keys']:
            formatted_text += "Foreign Keys:\n"
            for fk_col, fk_info in table_info['foreign_keys'].items():
                formatted_text += f"- {fk_col} references {fk_info['table']}.{fk_info['column']}\n"
            formatted_text += "\n"
        
        # Add column information
        formatted_text += "Columns:\n"
        for col_name, col_info in table_info['columns_info'].items():
            formatted_text += f"- {col_name} ({col_info['data_type']}): {col_info['description']}\n"
            
            # Add categorical value distribution if available and not too long
            if 'categorical_values' in col_info and len(col_info['categorical_values']) < 15:
                formatted_text += f"  Possible values: {', '.join(col_info['categorical_values'])}\n"
            
            # Add value range if available
            if 'value_range' in col_info:
                formatted_text += f"  Range: {col_info['value_range']['min']} to {col_info['value_range']['max']}\n"
        
        formatted_text += "\n"
        
        # Add important considerations
        if table_info['important_considerations']:
            formatted_text += f"Important Considerations: {table_info['important_considerations']}\n\n"
        
        # Add common joins
        if table_info['common_joins']:
            formatted_text += "Common Joins:\n"
            for join in table_info['common_joins']:
                formatted_text += f"- {join}\n"
            formatted_text += "\n"
        
        formatted_text += "---\n\n"
    
    return formatted_text

def create_llm_prompt(user_question, metadata_text):
    """Create the complete prompt to send to the LLM with improved instructions"""
    # Add explicit column information to the prompt
    prompt = f"""You are a professional medical database expert specializing in SQL and the MIMIC-IV database. Based on the user's question and the provided database metadata, generate a PostgreSQL query.

## User Question
{user_question}

## Database Metadata
{metadata_text}

## Task
1. Analyze the user question to determine which tables and columns need to be queried
2. Design an effective SQL query based on the provided metadata
3. Ensure the generated SQL is syntactically correct and considers table relationships
4. Use ONLY columns that are explicitly mentioned in the metadata for each table
5. If multiple table joins are needed, use the correct join conditions
6. Handle any potential edge cases
7. When dealing with medical codes (ICD diagnosis/procedure codes, medication codes), always join with their respective descriptor tables (d_icd_diagnoses, d_icd_procedures) to include both codes AND their human-readable descriptions
8. For medications, include the actual drug names from prescriptions.drug or emar.medication rather than just codes

## Response Format
Please return ONLY the SQL query without any explanation or comments. Start your answer with "SELECT" or "WITH" and end with a semicolon. Do not include anything else.

SQL Query:
"""
    return prompt

def generate_sql_with_openai(prompt):
    """Generate SQL query using OpenAI API with improved cleaning and validation"""
    try:
        # Using the new client format
        response = client.chat.completions.create(
            model="gpt-4",  # or another suitable model
            messages=[
                {"role": "system", "content": "You are a medical database expert who converts natural language questions into PostgreSQL queries. Return ONLY the SQL query with no explanations or comments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for more deterministic output
            max_tokens=500
        )
        
        # Get raw content
        raw_content = response.choices[0].message.content.strip()
        
        # More comprehensive cleaning of markdown and prefixes
        # Remove common prefixes
        prefixes = ["SQL Query:", "Query:", "PostgreSQL Query:"]
        for prefix in prefixes:
            if raw_content.startswith(prefix):
                raw_content = raw_content[len(prefix):].strip()
        
        # Remove markdown code blocks (handling various formats)
        import re
        sql_query = re.sub(r'```(?:sql|postgresql)?|```', '', raw_content)
        sql_query = sql_query.strip()
        
        # Find the termination point of the SQL part - look for typical SQL statement ending (semicolon) followed by a newline
        # This will remove explanatory text after the query
        match = re.search(r';[\s\n]*(\n|$)', sql_query)
        if match:
            # Only keep the part up to the semicolon
            sql_query = sql_query[:match.end()].strip()
        
        # Basic SQL syntax validation
        if not sql_query.lower().startswith(('select', 'with')):
            print("Warning: Generated SQL may not be valid. It doesn't start with SELECT or WITH.")
            
        # Log the cleaned query for debugging
        print(f"Cleaned SQL query: {sql_query[:100]}...")
        
        return sql_query
    
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None
    
def format_metadata_for_answer(metadata):
    """Format a simplified version of metadata for the answer generation prompt"""
    formatted_text = ""
    
    for table_name, table_info in metadata.items():
        formatted_text += f"Table: {table_name}\n"
        formatted_text += f"Description: {table_info['description']}\n\n"
        
        # Add key columns (simplified)
        formatted_text += "Key columns:\n"
        for col_name, col_info in table_info['columns_info'].items():
            if col_name in table_info.get('primary_keys', []) or 'key' in col_name.lower():
                formatted_text += f"- {col_name}: {col_info['description']}\n"
    
    return formatted_text

def generate_natural_language_answer(user_question, metadata, sql_query, query_results):
    """
    Generate a comprehensive natural language answer based on query results
    
    Parameters:
        user_question (str): Original user question
        metadata (dict): Metadata of relevant tables used in the query
        sql_query (str): Generated SQL query
        query_results: DataFrame or other format containing query results
        
    Returns:
        str: Natural language answer explaining the results
    """
    # Convert query results to a suitable format for LLM
    if isinstance(query_results, pd.DataFrame):
        # For large DataFrames, include sample and summary statistics
        if len(query_results) > 10:
            results_text = f"Results contain {len(query_results)} rows.\n\n"
            results_text += "Sample of first 5 rows:\n"
            results_text += query_results.head(5).to_string() + "\n\n"
            
            # Add summary statistics if numerical columns exist
            if any(query_results.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                results_text += "Summary statistics:\n"
                results_text += query_results.describe().to_string()
        else:
            results_text = "Results:\n" + query_results.to_string()
    else:
        results_text = f"Results: {str(query_results)}"
    
    # Create a comprehensive prompt for the LLM
    prompt = f"""
    User question: {user_question}
    
    Database information used:
    {format_metadata_for_answer(metadata)}
    
    SQL query executed:
    ```sql
    {sql_query}
    ```
    
    {results_text}
    
    Based on the above information, please provide a comprehensive answer to the user's question.
    Explain the results in natural language, highlighting key insights, patterns, or important values.
    If appropriate, suggest any follow-up analyses that might be valuable.
    """
    
    # Call the LLM to generate the answer
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that explains database query results clearly."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def attempt_fix_sql(sql_query, table_columns):
    """Attempt to automatically fix column reference issues in the SQL query"""
    prompt = f"""As a database expert, please fix the following SQL query to make it compatible with the provided table structures.
    
SQL query:
```sql
{sql_query}
```

Table structure information:
"""
    
    # Add table structure information
    for table, columns in table_columns.items():
        prompt += f"\nTable '{table}' columns: {', '.join(columns)}"
    
    prompt += """

Please provide the fixed SQL query, ensuring that:
1. Only use columns that exist in the tables
2. Fix any mismatches in join conditions
3. Maintain the basic logic and purpose of the query
4. Return only the fixed SQL query, without any explanations or comments

Fixed SQL query:
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an SQL expert, specializing in fixing errors in SQL queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        fixed_sql = response.choices[0].message.content.strip()
        
        # Clean formatting
        import re
        fixed_sql = re.sub(r'```(?:sql|postgresql)?|```', '', fixed_sql)
        fixed_sql = fixed_sql.strip()
        
        return fixed_sql
    
    except Exception as e:
        print(f"Error while attempting to fix SQL: {e}")
        return None
    

def classify_query(user_question, metadata):
    """
    Classify the query to identify if it's answerable with available data
    
    Parameters:
        user_question (str): The user's question
        metadata (dict): Metadata of most relevant tables
        
    Returns:
        dict: Classification results including status and reason
    """
    # Create a prompt for query classification
    tables_summary = "\n".join([f"- {table}: {info['description']}" 
                              for table, info in metadata.items()])
    
    prompt = f"""
    Based on the following database tables from a MEDICAL DATABASE (MIMIC-IV 2.2), 
    classify if this question can be answered:
    
    USER QUESTION: {user_question}
    
    AVAILABLE TABLES:
    {tables_summary}
    
    Please classify this question as one of:
    1. "answerable": Can be answered with the available tables
    2. "out_of_scope": Relates to medical data but not available in these tables
    3. "non_medical": Not related to medical data at all
    4. "future_data": Requires data from after the database collection period
    5. "private_data": Asks for personally identifiable information
    
    You must respond in valid JSON format with exactly these fields:
    {{
      "status": "one of the options above",
      "reason": "brief explanation why you classified it this way",
      "message": "user-friendly message explaining if/why the question can't be answered"
    }}
    """
    
    # Call LLM to classify, without specifying response_format
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies database queries. Always respond in valid JSON format."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Parse the JSON response
    import json
    try:
        classification = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        # Fallback in case the response isn't valid JSON
        return {
            "classification": "answerable",  # Default to answerable
            "reason": "Failed to parse classification response",
            "message": "I'll try to answer your question with the available data.",
            "status": "supported"
        }
    
    # Map the classification to pipeline control values
    result = {
        "classification": classification.get("status", "answerable"),
        "reason": classification.get("reason", "No reason provided"),
        "message": classification.get("message", "No message provided")
    }
    
    # Set overall status
    if result["classification"] == "answerable":
        result["status"] = "supported"
    else:
        result["status"] = "not_supported"
    
    return result