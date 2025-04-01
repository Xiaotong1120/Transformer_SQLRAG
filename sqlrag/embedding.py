import pandas as pd
import numpy as np
import json
from sqlrag.db_utils import get_db_connection
from sqlrag.llm import client
from config import SIMILARITY_TOP_K

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors
    
    Parameters:
        vec1 (list): First vector
        vec2 (list): Second vector
        
    Returns:
        float: Cosine similarity (between -1 and 1)
    """
    
    # Convert to numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)

def get_highest_similarity(metadata):
    """Extract the highest similarity score and corresponding table"""
    best_table = None
    best_score = 0
    
    for table_name, info in metadata.items():
        if "similarity_score" in info and info["similarity_score"] > best_score:
            best_score = info["similarity_score"]
            best_table = table_name
    
    return best_table, best_score


def vectorize_user_query(query_text):
    """
    Convert a user's natural language query into a vector representation
    
    Parameters:
        query_text (str): The user's natural language query
        
    Returns:
        list: Vector representation of the query
    """
    try:
        # Preprocess the query text - more processing steps can be added as needed
        processed_query = query_text.strip()
        
        # Generate embedding vector using OpenAI API
        response = client.embeddings.create(
            input=processed_query,
            model="text-embedding-3-small"  # Use the same model as for table embeddings
        )
        
        # Extract the embedding vector
        query_embedding = response.data[0].embedding
        
        print(f"✅ Successfully vectorized query: '{query_text[:50]}...' if len(query_text) > 50 else query_text")
        return query_embedding
    
    except Exception as e:
        print(f"❌ Error vectorizing query: {e}")
        return None
    
def fetch_relevant_metadata(query_embedding, top_k=SIMILARITY_TOP_K):
    """
    Fetch metadata for tables most relevant to the query embedding
    
    Parameters:
        query_embedding (list): Vector representation of the user query
        top_k (int): Number of most relevant tables to return
        
    Returns:
        dict: Dictionary mapping table names to their metadata with similarity scores
    """
    conn, cur = get_db_connection()
    
    try:
        print(f"Computing similarity to find top {top_k} relevant tables")
        cur.execute("""
            SELECT 
                table_name, description, table_purpose, columns_info, 
                primary_keys, foreign_keys, important_considerations,
                common_joins, example_questions, embedding
            FROM mimic_table_metadata
            WHERE embedding IS NOT NULL;
        """)
        
        rows = cur.fetchall()
        
        # Calculate similarity for each table
        table_similarities = []
        for row in rows:
            table_name = row[0]
            table_embedding = row[9]
            
            # Skip if embedding is NULL
            if table_embedding is None:
                continue
            
            # Convert string embedding to list of floats if needed
            if isinstance(table_embedding, str):
                import json
                table_embedding = json.loads(table_embedding.replace("'", '"'))
            
            # Calculate cosine similarity
            similarity = cosine_similarity(query_embedding, table_embedding)
            table_similarities.append((row, similarity))
        
        # Sort by similarity (descending) and take top_k
        table_similarities.sort(key=lambda x: x[1], reverse=True)
        top_tables = table_similarities[:top_k]
        
        # Format as dictionary
        tables_metadata = {}
        for row, similarity in top_tables:
            table_name = row[0]
            tables_metadata[table_name] = {
                'description': row[1],
                'table_purpose': row[2],
                'columns_info': row[3],
                'primary_keys': row[4],
                'foreign_keys': row[5],
                'important_considerations': row[6],
                'common_joins': row[7],
                'example_questions': row[8],
                'similarity_score': similarity  # Add similarity score to metadata
            }
            
            # Print similarity for debugging
            print(f"Table: {table_name}, Similarity: {similarity:.4f}")
        
        return tables_metadata
    
    finally:
        cur.close()
        conn.close()