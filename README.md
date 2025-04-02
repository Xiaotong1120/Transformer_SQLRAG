# SQL Retrieval-Augmented Generation Pipeline for MIMIC-IV Hospital Data

## Project Overview and Motivation

This project implements a **retrieval-augmented generation (RAG)** pipeline that allows users to ask natural language questions and receive answers from a **medical relational database**. The system is built around a subset of the **MIMIC-IV v2.2 “hosp” module** (a large critical care database), enabling complex queries about hospital patients, diagnoses, lab results, etc., without writing SQL manually. The motivation is to **bridge the gap between clinicians/researchers and raw data** – formulating SQL queries on MIMIC-IV requires detailed knowledge of the schema and medical coding, which many users lack. By using RAG, we leverage a combination of a vector database and GPT-4 to translate plain English questions into valid SQL and then explain the results. This approach has been shown to significantly improve accuracy over prompting an LLM alone, and it mitigates hallucinations by grounding answers in actual data.

**Why Retrieval-Augmented Generation (RAG)?** In a RAG setup, the system first retrieves relevant context (in this case, database table information) and provides it to the language model to guide generation. Here we treat **table descriptions** as the knowledge to retrieve, rather than documents. GPT-4 is then used to generate a SQL query that exactly targets the needed data, ensuring the answer is based on **real patient data** and not the model’s memory. This design harnesses the strengths of both databases and LLMs: the database provides factual, up-to-date data, and the LLM provides flexibility in understanding natural language. It also helps preserve data integrity – rather than trying to stuff a huge medical database into the LLM’s context, we only supply the schema info needed for a given question. This **reduces errors and protects patient privacy**, since only schema metadata and aggregated results (no raw identifiers) are exposed to the LLM.

Overall, the project demonstrates how clinicians or analysts can **query the MIMIC-IV database by simply asking questions** (e.g., “How many patients over 80 were diagnosed with sepsis?”) and get an answer with an explanation. This can accelerate exploratory data analysis and hypothesis generation, empowering users who are not SQL experts to interact with complex healthcare datasets.

---

## System Architecture and Workflow

The system consists of an **offline ingestion phase** and an **online query-answering phase**:

- **Offline (Preprocessing/Ingestion):**  
  We curate a metadata table containing documentation for each table in the MIMIC-IV subset. This includes the table’s description, purpose, column info (with data types and meanings), primary/foreign keys, common join relationships, example questions, and synonyms. Each table’s metadata is then encoded into a vector embedding using OpenAI’s embedding model, and stored in Postgres (using the `VECTOR` data type from pgVector). This forms our *vector index* of knowledge. Essentially, we index the schema documentation so the system can semantically search which tables might be relevant to a new question.

- **Online (Query Workflow):**  
  When a user poses a question through the interface, the system goes through a sequence of stages to produce the answer. At a high level:

  1. **Embed the Question**: Convert the user’s query into a vector (via OpenAI’s `text-embedding-3-small` model).
  2. **Retrieve Relevant Tables**: Compare this query vector to table metadata vectors stored in Postgres, returning the top-k most relevant tables.
  3. **Classify Query**: Ensure the question is “answerable” and not disallowed or out of scope.
  4. **LLM Prompting**: Provide GPT-4 with the relevant table metadata, plus guidelines, to generate a **PostgreSQL** query.
  5. **SQL Validation & Fixing**: Check that the generated SQL references valid tables/columns; if it’s incorrect, GPT-4 tries to fix it.
  6. **Execute the Query**: Run the validated SQL against the MIMIC-IV database in Postgres.
  7. **Summarize Results**: GPT-4 is then prompted again to produce a user-friendly explanation of the results.

---

### 1. Query Embedding & Table Retrieval
When a question comes in (e.g. “Which diagnoses are most common among elderly patients?”), the first step is to vectorize the user’s query. We use OpenAI’s latest text embedding model (the "text-embedding-3-small" model, 1536 dimensions) to convert the question into an embedding vector. This numeric representation captures the semantic meaning of the question, allowing us to compare it with the embeddings of table metadata. The choice of OpenAI’s embedding model is due to its strong performance and efficiency – text-embedding-3-small significantly outperforms the older ada-002 model on retrieval tasks (44% vs 31% on a multilingual info retrieval benchmark) while being 5× cheaper​. With the query vector in hand, we perform a cosine similarity search against the stored table embeddings to find which tables are most related to the question. This is implemented using the pgVector extension in PostgreSQL, which allows storing embeddings and performing similarity queries directly in SQL​. (In our code, we simply fetch all table embeddings and compute cosine similarity in Python for simplicity, but this could be done with a ORDER BY embedding <-> query_vector LIMIT k SQL query as well.) The top K tables by similarity score are returned as the relevant schema context. For example, if the question mentioned “diagnoses” and “patients older than X”, the retrieval might return the patients table and the diagnoses_icd table metadata as the most relevant. Each retrieved table comes with its stored metadata (description, columns, keys, etc.). By using embeddings, we capture semantic matches – the user’s phrasing doesn’t have to exactly match table names. For instance, someone asking about “medications given to patients” would semantically match the emar (electronic medication administration record) table even if they didn’t use the word "emar", because the metadata for emar includes terms like “medication” and “administration”. This vector-based retrieval is far more robust than simple keyword mapping, which might miss relevant tables if synonyms or implicit concepts are used.

### 2. Relevance Filtering and Query Classification
After retrieval, the system examines the results to ensure they make sense before proceeding:
Similarity Threshold Check: We look at the highest similarity score among the retrieved tables. If the best match is below a certain threshold (configured as SIMILARITY_THRESHOLD, e.g. 0.2), it indicates the user’s question might not actually pertain to any data we have. In that case, the pipeline will return a friendly message like “I don’t have the necessary data to answer that question.” rather than attempting a nonsensical query. This prevents the system from going off-track when faced with an unrelated query.
Query Classification: We then perform a classification of the user’s query to determine if it’s answerable with our data and whether it’s appropriate to proceed. This uses GPT-4 in a zero-shot manner. We provide GPT-4 with a summary of the available tables (the names and descriptions of the top retrieved tables) and ask it to classify the question into categories such as "answerable", "out_of_scope", "non_medical", "future_data", or "private_data". This is essentially a safety and relevance check:
Out of scope might be a medical question that our database doesn’t cover (e.g. a question about genetic data when we only have hospital records).Non-medical would be a query completely unrelated to healthcare.
Future data implies the question needs data beyond the collection period (MIMIC-IV covers 2008–2019​, so asking about COVID-19 trends, for example, is future/out of range).
Private data flags any query asking for personally identifiable info or something that could violate patient privacy (even though MIMIC-IV is de-identified, we ensure we don’t facilitate re-identification attempts).
This classification step is important for ethical considerations. If GPT-4 determines the question is not answerable or not appropriate, the pipeline stops and returns an explanatory message to the user (e.g. “I’m sorry, I cannot answer that because it asks for personal patient information.”). This ensures we do not produce SQL for disallowed queries. In practice, most general analytical questions will be "answerable" and the pipeline will continue.

### 3. Prompt Construction with Retrieved Metadata
If the query is classified as answerable, we move on to preparing the prompt for SQL generation. We take the metadata of the retrieved tables and format it into a schema summary that will be given to GPT-4. This formatted metadata includes:
  Table name and description: A brief explanation of what each table contains.
  Table purpose: (if provided) the intended use or key info of the table.
  Columns: Every column name, its data type, and a short description or meaning. We might also include known value ranges or example values (e.g. for categorical codes).
  Primary keys and Foreign keys: So the model knows how tables relate (e.g. diagnoses_icd has foreign keys linking to patients via subject_id and to a dictionary table for code meanings).
  Important considerations: Any caveats (e.g. certain columns might be sparsely populated or only apply to certain patient types).
  Common joins: Typical ways this table is joined with others (for example, diagnoses_icd commonly joins with d_icd_diagnoses to get the diagnosis description).
  Synonyms or terms: Alternate names or abbreviations for concepts in the table (to help the embedding and also possibly the LLM if the user used a different term).
By assembling this rich context, we essentially give GPT-4 a mini documentation of the relevant part of the database schema. This is critical for guiding it to write correct SQL. The system concatenates the metadata for all top-K tables into one prompt section. We also explicitly instruct GPT-4 on the task: “You are a professional medical database expert. Given the user question and the provided database metadata, generate a PostgreSQL query.” The prompt then enumerates specific guidelines for the LLM, for example:

  1. Analyze the user question to determine which tables and columns are needed.
  2. Design an effective SQL query based on the provided metadata.
  3. Ensure the SQL is syntactically correct and uses proper table relationships.
  4. Use ONLY columns that are mentioned in the metadata for each table.
  5. If multiple tables need to be joined, use the correct join keys.
  6. Handle any potential edge cases.
  7. When dealing with medical codes (ICD codes, etc.), join with their descriptor tables to get human-readable descriptions.
  8. For medication records, include the drug name rather than just codes.
These instructions (as seen in our create_llm_prompt function) encapsulate domain-specific knowledge (like always joining on code description tables such as d_icd_diagnoses for diagnoses) and enforce a clean output format. They were crafted based on our design goals: we want the generated SQL to be correct, complete, and independent (self-contained), and we don’t want the model to include any explanatory text (since we just need the query). By structuring the prompt with bullet points and constraints, we reduce the chances of GPT-4 producing invalid or partial SQL. This prompt engineering is a key design decision – it acts like a blueprint that GPT-4 must follow, effectively transferring some schema expertise to the model through instructions.
   
