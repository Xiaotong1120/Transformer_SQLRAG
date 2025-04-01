import streamlit as st
import pandas as pd
import numpy as np
import time
from sqlrag.pipeline import sqlrag_pipeline

# Set page title and configuration
st.set_page_config(
    page_title="Medical Database Query Assistant",
    page_icon="🏥",
    layout="wide"
)

# Page title
st.title("Natural Language Query System for Medical Database")
st.markdown("### Query the MIMIC-IV medical database using natural language")

# Introduction
with st.expander("About this System", expanded=False):
    st.markdown("""
    This system allows you to query the MIMIC-IV medical database using natural language. It will:
    1. Analyze your question
    2. Find relevant data tables
    3. Generate appropriate SQL queries
    4. Execute the query and return results
    5. Explain the results in simple language
    
    Please use clear medical-related questions with enough details for the system to understand your intent.
    """)

# User input area
user_question = st.text_area("Enter your question:", 
                            height=100, 
                            placeholder="For example: 'Which patients stayed in the ICU for more than 7 days and were diagnosed with sepsis?'")

# Submit button
if st.button("Submit Query"):
    if user_question:
        # Show processing status
        with st.spinner("Processing your query..."):
            # Record start time
            start_time = time.time()
            
            # Call pipeline to process query
            result = sqlrag_pipeline(user_question)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Check for errors
            if "error" in result:
                st.error(f"Error processing query: {result['error']}")
                st.info(result.get("answer", "Please try rephrasing your question."))
            else:
                # Display results
                st.success(f"Query processed successfully! (in {processing_time:.2f} seconds)")
                
                # Display answer
                st.markdown("### Answer")
                st.markdown(result["answer"])
                
                # Display generated SQL query
                with st.expander("View Generated SQL Query"):
                    st.code(result["generated_sql"], language="sql")
                
                # Display query results data
                if "results" in result and not result["results"].empty:
                    with st.expander("View Raw Query Results"):
                        st.dataframe(result["results"])
    else:
        st.warning("Please enter a question to query.")

# Footer
st.markdown("---")
st.markdown("© 2025 Medical Database Query Assistant | Based on MIMIC-IV Database")