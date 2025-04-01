import pandas as pd
import numpy as np
from sqlrag.embedding import vectorize_user_query, fetch_relevant_metadata, get_highest_similarity
from sqlrag.llm import format_metadata_for_prompt, create_llm_prompt, generate_sql_with_openai
from sqlrag.llm import attempt_fix_sql, classify_query, generate_natural_language_answer
from sqlrag.db_utils import execute_sql_query, check_query_columns
from config import SIMILARITY_THRESHOLD, SIMILARITY_TOP_K

def sqlrag_pipeline(user_question):
    """Execute the complete SQLRAG pipeline with enhanced safety checks"""
    print(f"User Question: {user_question}")
    
    # 1. Vectorize user query
    print("Vectorizing user query...")
    query_embedding = vectorize_user_query(user_question)
    
    if not query_embedding:
        return {"error": "Failed to vectorize user query"}
    
    # 2. Fetch relevant metadata using vector similarity
    print("Finding relevant tables...")
    metadata = fetch_relevant_metadata(query_embedding, top_k=SIMILARITY_TOP_K)
    
    # 3. Check if any relevant tables were found with good similarity
    if not metadata:
        return {
            "user_question": user_question,
            "error": "No relevant tables found in the database for this query.",
            "answer": "I don't have the necessary data to answer this question. The database doesn't contain information related to your query."
        }
    
    # 4. Check similarity scores to ensure they're above threshold
    # Get the highest similarity score
    best_table, best_similarity = get_highest_similarity(metadata)
    if best_similarity < SIMILARITY_THRESHOLD:  # Adjust threshold as needed
        return {
            "user_question": user_question,
            "best_match": best_table,
            "similarity": best_similarity,
            "error": "The query doesn't seem to match well with available data.",
            "answer": f"Your question might not be answerable with the available medical data. The closest match I found was related to '{best_table}' but the relevance is low."
        }
    
    # 5. Use query classifier to identify query intent and feasibility
    query_classification = classify_query(user_question, metadata)
    if query_classification["status"] == "not_supported":
        return {
            "user_question": user_question,
            "error": query_classification["reason"],
            "answer": query_classification["message"]
        }
    
    # 6. Format metadata for prompt - ensure metadata includes complete column information
    metadata_text = format_metadata_for_prompt(metadata)
    
    # 7. Create LLM prompt with improved instructions
    prompt = create_llm_prompt(user_question, metadata_text)
    
    print("Generating SQL with LLM...")
    # 8. Generate SQL query
    sql_query = generate_sql_with_openai(prompt)
    
    if not sql_query:
        return {
            "user_question": user_question,
            "error": "Failed to generate SQL query",
            "answer": "I couldn't generate a SQL query to answer your question. Please try rephrasing it."
        }
    
    print(f"Generated SQL: \n{sql_query}\n")
    
    # 9. New step: Validate the generated SQL against table structure
    print("Validating SQL against database structure...")
    issues, table_columns = check_query_columns(sql_query)
    
    if issues:
        print("Potential issues detected:")
        for issue in issues:
            print(f"  - {issue}")
        
        # Try to automatically fix the SQL
        print("Attempting to fix SQL...")
        fixed_sql = attempt_fix_sql(sql_query, table_columns)
        
        if fixed_sql:
            print(f"Fixed SQL: \n{fixed_sql}\n")
            sql_query = fixed_sql
        else:
            return {
                "user_question": user_question, 
                "error": f"Generated SQL query is incompatible with database structure: {'; '.join(issues)}",
                "generated_sql": sql_query,
                "answer": "I couldn't generate a valid query compatible with the database structure. There might be a mismatch in my understanding of the database schema."
            }
    
    print("Executing SQL query...")
    # 10. Execute SQL query
    results = execute_sql_query(sql_query)
    
    if results is None:
        return {
            "user_question": user_question,
            "error": "Error executing SQL query",
            "generated_sql": sql_query,
            "answer": "I apologize, but I couldn't execute the generated SQL query. There might be an issue with the database or the query structure."
        }
    
    print("Generating natural language answer...")
    # 11. Generate natural language answer
    answer = generate_natural_language_answer(user_question, metadata, sql_query, results)
    
    return {
        "user_question": user_question,
        "generated_sql": sql_query,
        "results": results,
        "answer": answer
    }