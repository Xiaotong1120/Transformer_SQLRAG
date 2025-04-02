# SQL Retrieval-Augmented Generation Pipeline for MIMIC-IV Hospital Data

## Table of Contents
- [Project Overview and Motivation](#project-overview-and-motivation)
- [System Architecture and Workflow](#system-architecture-and-workflow)
  - [1. Query Embedding & Table Retrieval](#1-query-embedding--table-retrieval)
  - [2. Relevance Filtering and Query Classification](#2-relevance-filtering-and-query-classification)
  - [3. Prompt Construction with Retrieved Metadata](#3-prompt-construction-with-retrieved-metadata)
  - [4. SQL Generation with GPT-4](#4-sql-generation-with-gpt-4)
  - [5. SQL Validation and Automatic Correction](#5-sql-validation-and-automatic-correction)
  - [6. Query Execution on the Database](#6-query-execution-on-the-database)
  - [7. Results Summarization in Natural Language](#7-results-summarization-in-natural-language)
- [Technologies and Models Used](#technologies-and-models-used)
- [Dataset Details and Ethical Considerations](#dataset-details-and-ethical-considerations)
- [Setup and Installation](#setup-and-installation)
- [Design Rationale and Discussion](#design-rationale-and-discussion)
- [Possible Extensions and Future Work](#possible-extensions-and-future-work)
- [Project Structure](#project-structure)

## Project Overview and Motivation

This project implements a **retrieval-augmented generation (RAG)** pipeline that allows users to ask natural language questions and receive answers from a **medical relational database**. The system is built around a subset of the **MIMIC-IV v2.2 "hosp" module** (a large critical care database), enabling complex queries about hospital patients, diagnoses, lab results, etc., without writing SQL manually. The motivation is to **bridge the gap between clinicians/researchers and raw data** – formulating SQL queries on MIMIC-IV requires detailed knowledge of the schema and medical coding, which many users lack. By using RAG, we leverage a combination of a vector database and GPT-4 to translate plain English questions into valid SQL and then explain the results. This approach has been shown to significantly improve accuracy over prompting an LLM alone, and it mitigates hallucinations by grounding answers in actual data.

**Why Retrieval-Augmented Generation (RAG)?** In a RAG setup, the system first retrieves relevant context (in this case, database table information) and provides it to the language model to guide generation. Here we treat **table descriptions** as the knowledge to retrieve, rather than documents. GPT-4 is then used to generate a SQL query that exactly targets the needed data, ensuring the answer is based on **real patient data** and not the model's memory. This design harnesses the strengths of both databases and LLMs: the database provides factual, up-to-date data, and the LLM provides flexibility in understanding natural language. It also helps preserve data integrity – rather than trying to stuff a huge medical database into the LLM's context, we only supply the schema info needed for a given question. This **reduces errors and protects patient privacy**, since only schema metadata and aggregated results (no raw identifiers) are exposed to the LLM.

Overall, the project demonstrates how clinicians or analysts can **query the MIMIC-IV database by simply asking questions** (e.g., "How many patients over 80 were diagnosed with sepsis?") and get an answer with an explanation. This can accelerate exploratory data analysis and hypothesis generation, empowering users who are not SQL experts to interact with complex healthcare datasets.

## System Architecture and Workflow

The system consists of an **offline ingestion phase** and an **online query-answering phase**:

### Offline (Preprocessing/Ingestion)

We curate a metadata table containing documentation for each table in the MIMIC-IV subset. This includes the table's description, purpose, column info (with data types and meanings), primary/foreign keys, common join relationships, example questions, and synonyms. Each table's metadata is then encoded into a vector embedding using OpenAI's embedding model, and stored in Postgres (using the `VECTOR` data type from pgVector). This forms our *vector index* of knowledge. Essentially, we index the schema documentation so the system can semantically search which tables might be relevant to a new question.

### Online (Query Workflow)

When a user poses a question through the interface, the system goes through a sequence of stages to produce the answer. At a high level:

1. **Embed the Question**: Convert the user's query into a vector (via OpenAI's `text-embedding-3-small` model).
2. **Retrieve Relevant Tables**: Compare this query vector to table metadata vectors stored in Postgres, returning the top-k most relevant tables.
3. **Classify Query**: Ensure the question is "answerable" and not disallowed or out of scope.
4. **LLM Prompting**: Provide GPT-4 with the relevant table metadata, plus guidelines, to generate a **PostgreSQL** query.
5. **SQL Validation & Fixing**: Check that the generated SQL references valid tables/columns; if it's incorrect, GPT-4 tries to fix it.
6. **Execute the Query**: Run the validated SQL against the MIMIC-IV database in Postgres.
7. **Summarize Results**: GPT-4 is then prompted again to produce a user-friendly explanation of the results.

### 1. Query Embedding & Table Retrieval

When a question comes in (e.g. "Which diagnoses are most common among elderly patients?"), the first step is to vectorize the user's query. We use OpenAI's latest text embedding model (the "text-embedding-3-small" model, 1536 dimensions) to convert the question into an embedding vector. This numeric representation captures the semantic meaning of the question, allowing us to compare it with the embeddings of table metadata.

The choice of OpenAI's embedding model is due to its strong performance and efficiency – text-embedding-3-small significantly outperforms the older ada-002 model on retrieval tasks (44% vs 31% on a multilingual info retrieval benchmark) while being 5× cheaper.

With the query vector in hand, we perform a cosine similarity search against the stored table embeddings to find which tables are most related to the question. This is implemented using the pgVector extension in PostgreSQL, which allows storing embeddings and performing similarity queries directly in SQL. (In our code, we simply fetch all table embeddings and compute cosine similarity in Python for simplicity, but this could be done with a ORDER BY embedding <-> query_vector LIMIT k SQL query as well.) The top K tables by similarity score are returned as the relevant schema context.

For example, if the question mentioned "diagnoses" and "patients older than X", the retrieval might return the patients table and the diagnoses_icd table metadata as the most relevant. Each retrieved table comes with its stored metadata (description, columns, keys, etc.).

By using embeddings, we capture semantic matches – the user's phrasing doesn't have to exactly match table names. For instance, someone asking about "medications given to patients" would semantically match the emar (electronic medication administration record) table even if they didn't use the word "emar", because the metadata for emar includes terms like "medication" and "administration". This vector-based retrieval is far more robust than simple keyword mapping, which might miss relevant tables if synonyms or implicit concepts are used.

### 2. Relevance Filtering and Query Classification

After retrieval, the system examines the results to ensure they make sense before proceeding:

**Similarity Threshold Check**: We look at the highest similarity score among the retrieved tables. If the best match is below a certain threshold (configured as SIMILARITY_THRESHOLD, e.g. 0.2), it indicates the user's question might not actually pertain to any data we have. In that case, the pipeline will return a friendly message like "I don't have the necessary data to answer that question." rather than attempting a nonsensical query. This prevents the system from going off-track when faced with an unrelated query.

**Query Classification**: We then perform a classification of the user's query to determine if it's answerable with our data and whether it's appropriate to proceed. This uses GPT-4 in a zero-shot manner. We provide GPT-4 with a summary of the available tables (the names and descriptions of the top retrieved tables) and ask it to classify the question into categories such as:
- "answerable"
- "out_of_scope"
- "non_medical"
- "future_data"
- "private_data"

This is essentially a safety and relevance check:
- **Out of scope** might be a medical question that our database doesn't cover (e.g. a question about genetic data when we only have hospital records).
- **Non-medical** would be a query completely unrelated to healthcare.
- **Future data** implies the question needs data beyond the collection period (MIMIC-IV covers 2008–2019, so asking about COVID-19 trends, for example, is future/out of range).
- **Private data** flags any query asking for personally identifiable info or something that could violate patient privacy (even though MIMIC-IV is de-identified, we ensure we don't facilitate re-identification attempts).

This classification step is important for ethical considerations. If GPT-4 determines the question is not answerable or not appropriate, the pipeline stops and returns an explanatory message to the user (e.g. "I'm sorry, I cannot answer that because it asks for personal patient information."). This ensures we do not produce SQL for disallowed queries. In practice, most general analytical questions will be "answerable" and the pipeline will continue.

### 3. Prompt Construction with Retrieved Metadata

If the query is classified as answerable, we move on to preparing the prompt for SQL generation. We take the metadata of the retrieved tables and format it into a schema summary that will be given to GPT-4. This formatted metadata includes:

- **Table name and description**: A brief explanation of what each table contains.
- **Table purpose**: (if provided) the intended use or key info of the table.
- **Columns**: Every column name, its data type, and a short description or meaning. We might also include known value ranges or example values (e.g. for categorical codes).
- **Primary keys and Foreign keys**: So the model knows how tables relate (e.g. diagnoses_icd has foreign keys linking to patients via subject_id and to a dictionary table for code meanings).
- **Important considerations**: Any caveats (e.g. certain columns might be sparsely populated or only apply to certain patient types).
- **Common joins**: Typical ways this table is joined with others (for example, diagnoses_icd commonly joins with d_icd_diagnoses to get the diagnosis description).
- **Synonyms or terms**: Alternate names or abbreviations for concepts in the table (to help the embedding and also possibly the LLM if the user used a different term).

By assembling this rich context, we essentially give GPT-4 a mini documentation of the relevant part of the database schema. This is critical for guiding it to write correct SQL. The system concatenates the metadata for all top-K tables into one prompt section.

We also explicitly instruct GPT-4 on the task: "You are a professional medical database expert. Given the user question and the provided database metadata, generate a PostgreSQL query." The prompt then enumerates specific guidelines for the LLM, for example:

1. Analyze the user question to determine which tables and columns are needed.
2. Design an effective SQL query based on the provided metadata.
3. Ensure the SQL is syntactically correct and uses proper table relationships.
4. Use ONLY columns that are mentioned in the metadata for each table.
5. If multiple tables need to be joined, use the correct join keys.
6. Handle any potential edge cases.
7. When dealing with medical codes (ICD codes, etc.), join with their descriptor tables to get human-readable descriptions.
8. For medication records, include the drug name rather than just codes.

These instructions (as seen in our create_llm_prompt function) encapsulate domain-specific knowledge (like always joining on code description tables such as d_icd_diagnoses for diagnoses) and enforce a clean output format. They were crafted based on our design goals: we want the generated SQL to be correct, complete, and independent (self-contained), and we don't want the model to include any explanatory text (since we just need the query).

By structuring the prompt with bullet points and constraints, we reduce the chances of GPT-4 producing invalid or partial SQL. This prompt engineering is a key design decision – it acts like a blueprint that GPT-4 must follow, effectively transferring some schema expertise to the model through instructions.

### 4. SQL Generation with GPT-4

Given the user question and the schema context prompt, we call OpenAI's GPT-4 model (via the Chat Completion API) to generate the SQL query. We set the model to a low temperature (e.g. 0.1) to minimize randomness, favoring a deterministic, optimal solution each time.

GPT-4 then responds with a block of SQL text. We post-process its output to ensure it's clean:
- We strip any markdown formatting (sometimes the model might return the query inside sql code blocks, which we remove).
- We remove any extraneous prefixes like "SQL Query:" that might sneak in, enforcing that the output starts directly with SELECT or WITH.
- We also ensure it ends with a proper semicolon.

After cleaning, we have the generated SQL query as a string. For example, for the earlier hypothetical question about common diagnoses in elderly patients, GPT-4 might produce something like:

```sql
SELECT d.long_title AS diagnosis, COUNT(*) AS patient_count
FROM diagnoses_icd di
JOIN patients p ON di.subject_id = p.subject_id
JOIN d_icd_diagnoses d ON di.icd_code = d.icd_code AND di.icd_version = d.icd_version
WHERE p.anchor_age > 80
GROUP BY d.long_title
ORDER BY patient_count DESC;
```

This step relies on GPT-4's ability to understand the question and the provided schema context to form a valid query. GPT-4 was chosen because of its superior capability in understanding complex instructions and generating syntactically correct and semantically appropriate SQL. Simpler models (like GPT-3.5) were less reliable for this domain, often misunderstanding the schema or missing join conditions. Research in 2024 has shown that integrating retrieval context (as we do) helps LLMs generate much better SQL for EHR data, and we observed GPT-4's outputs to be quite good in initial testing.

### 5. SQL Validation and Automatic Correction

Even with GPT-4's prowess, there is still a risk that the generated SQL might not be perfectly compatible with our database. The model could, for instance, use a wrong column name (perhaps a typo or using an alias incorrectly), or join to a table that wasn't among the retrieved ones due to hallucination.

To catch such issues before running the query, the pipeline includes a SQL validation step. We wrote a utility (check_query_columns) that parses the SQL string to identify all table names and column references used. It then checks each referenced table and column against the actual database schema:
- If a table or column doesn't exist in our database, we flag it as an issue.
- We gather all columns of each table (via a simple query to the information schema or a cached schema dict) to aid in this validation.

If any issues are found (e.g. "Column XYZ does not exist in table patients"), we do not directly execute the faulty SQL. Instead, we attempt an automatic fix using GPT-4. We prompt GPT-4 with a message that lists the SQL it gave and the schema of the involved tables, asking it to correct the query to use the right columns or joins.

Essentially, the prompt says: "Please fix the following SQL query to align with the provided table structures. Here are the columns in each relevant table: ... (list columns). Ensure you only use existing columns and correct any join keys or naming issues. Return only the fixed SQL." This uses GPT-4 in a specialized role as an SQL repair expert. Since it now knows exactly which columns exist, it can adjust the query accordingly.

For example, if the original query used p.age but the patients table actually uses anchor_age, the fix step will correct that. If GPT-4 returns a fixed query, we replace the original with the fixed version and proceed. In many cases, this automated correction can save the pipeline from failing due to minor mistakes.

If the query is beyond repair or GPT-4 cannot fix it (this is rare), we would then return an error to the user indicating that we couldn't generate a valid query.

Why validate? Without validation, a bad SQL query would either error out at execution time or, worse, run but yield incorrect results. By catching mistakes early, we maintain trust in the system's output. It's worth noting that our validation is somewhat simplistic (regex-based and not a full SQL parser), so it may not catch every possible issue, but it covers the common structural problems like missing table or column names.

This approach of a "try, validate, and repair" loop makes the system more robust, essentially giving GPT-4 two chances to get it right (initial generation, then correction if needed).

### 6. Query Execution on the Database

Once we have a validated SQL query, the system executes it against the PostgreSQL database (which holds the MIMIC-IV subset). We use a Python DB API (psycopg2) with a safe execution function that applies a timeout (e.g. 30 seconds) to avoid long-running queries and a row limit (e.g. 50,000 rows) to avoid overwhelming the user or the memory. The query is run in a read-only transaction.

Thanks to the earlier steps, by the time we run it, this SQL is expected to be correct and relevant. On execution, there are a few possibilities:
- For aggregate queries or selective queries, we get a result set (which could be a few rows to thousands of rows). We fetch the results into a Pandas DataFrame for ease of manipulation.
- If the query was something like just creating a temporary table or had no output (which in our use case shouldn't happen, since we focus on SELECT queries for answering questions), we handle that separately, but typically we always do a SELECT to answer the user's question.
- If any database error occurs at this stage (e.g. a rare case where the query is syntactically correct but fails for some reason, or network issues connecting to DB), we catch it and return an error message to the user. This is an extra safety net, though our prior validation makes it unlikely for execution to fail.

### 7. Results Summarization in Natural Language

Finally, we have the raw results – often a table of numbers or text. While the user could examine a data table, our system aims to provide a concise natural language answer. We use GPT-4 one more time to summarize and explain the query results in a user-friendly way.

We prepare a prompt for GPT-4 that includes:
- A restatement of the original question (for context).
- A brief recap of the data context (which tables were used, from the metadata, just to remind the model of the meaning of columns).
- The SQL query that was run (so the model knows exactly what was asked of the data).
- The results of the query. We format the results as a small snippet:
  - If the result set is large (more than a certain number of rows, say >10), we don't feed everything (to avoid hitting token limits). Instead, we might provide the number of rows and perhaps the first few rows as a sample, plus some summary statistics if applicable (like mean, min, max for numeric columns). For example, "Results contain 500 rows. Here is a sample of 5 rows: ... And here are some summary stats: ...".
  - If the result set is small (just a few rows or a single row), we can include it entirely in a tabular text form.

Then we instruct GPT-4 to "Provide a comprehensive answer to the user's question based on these results. Explain in plain language and highlight key insights or patterns. If appropriate, mention any follow-up questions or analyses." We set the role to a helpful explainer.

GPT-4 then generates a paragraph or two that answers the question. For example, if the question was "Which diagnoses are most common among elderly patients?", GPT-4 might say:

"Patients above 80 years old were most frequently diagnosed with Hypertension, followed by Chronic Kidney Disease and Diabetes Mellitus. In fact, Hypertension was present in 45% of these patients. Chronic Kidney Disease was the second most common, affecting about 30%. Diabetes was third at 25%. These conditions are common age-related illnesses, which explains their high prevalence in the elderly population. Other diagnoses in the top 10 included coronary artery disease and pneumonia. This suggests that cardiovascular and respiratory conditions are prominent in the elderly demographic. Further analysis could look at outcomes for these patients or the distribution by gender."

(This is a hypothetical answer, but illustrates how the model turns the raw counts into an explanatory narrative.)

This final answer is what the user sees as the main output. We also display the generated SQL and optionally the raw data table (for transparency, via collapsible sections in the Streamlit app), but the focus is on the natural language explanation, since the goal is to make the data accessible.

By using GPT-4 to explain the results, we ensure the answer is not just a number or table but contextualized with medical insight. GPT-4's language capabilities shine here, as it can mention clinical relevance (as in the hypothetical example, noting conditions are age-related).

Of course, we must caution that GPT-4 is not a doctor – it provides a summary, but users should interpret the results in proper clinical context. The model might sometimes over-generalize or see patterns that are obvious from data but require careful interpretation (we rely on the user, likely a researcher, to critically assess the answer).

In summary, the online workflow goes through embedding → retrieval → GPT-4 (SQL) → validation → execution → GPT-4 (summary). The entire pipeline is orchestrated by the sqlrag_pipeline() function in our code, which glues these components together and handles errors at each step to return a final response dictionary.

## Technologies and Models Used

This project brings together a range of technologies – from database extensions to large language models – each chosen for a specific purpose:

### PostgreSQL 15 + pgVector
We use PostgreSQL as the core database to store the MIMIC-IV data. The pgVector extension allows us to treat Postgres as a vector database in addition to a relational database. By storing table embeddings in Postgres, we keep all data and metadata in one place, simplifying deployment. pgVector supports indexing and efficient approximate nearest neighbor search on vectors, meaning our design can scale to many tables or even documents without significant slowdown.

We chose pgVector over an external vector store (like Pinecone or FAISS) to reduce system complexity – no separate service is needed, and the vector search happens right where the data lives. This tight integration is powerful: for instance, one could write a single SQL query joining vector similarity results with actual data if needed.

The rest of the MIMIC data is in standard relational tables (with primary keys, foreign keys, etc.), and Postgres reliably handles the execution of the complex SQL that GPT-4 generates.

### OpenAI Embedding Model (text-embedding-3-small)
This is the model used to convert text to embeddings. We selected it for its balance of performance and cost. According to OpenAI, text-embedding-3-small is a newer embedding model that provides a substantial performance boost over the previous generation (Ada) on retrieval tasks, and it operates at 1/5th the cost.

The embedding dimension (1536) captures a rich semantic representation of texts. We use it both to embed table metadata (during setup) and each user query at runtime. Consistency is important – using the same model for both ensures the vectors live in the same semantic space.

If needed, one could use an open-source embedding model (like SentenceTransformers or similar) to avoid external API calls, but we opted for OpenAI's model for convenience and quality. (All embedding calls and LLM calls use the OpenAI API, with the key set in an environment variable.)

### OpenAI GPT-4
GPT-4 is the backbone of the generation steps – it's used in three places: to generate the SQL, to classify the query, and to summarize the results (and also for SQL fixes).

GPT-4 was chosen because of its superior capability in understanding complex instructions and handling domain-specific content. Medical databases have lots of jargon (e.g., abbreviations like DRG or ICD codes); GPT-4 has been found to have stronger proficiency with such content compared to earlier models. Also, writing SQL for complex schema requires reasoning about joins and filters – GPT-4's larger context window and reasoning ability make it much more reliable for this task.

Prior works have noted that GPT-4 outperforms 3.5 in text-to-SQL tasks, especially when information needs to be retrieved or when following a structured prompt. We run GPT-4 in "chat" mode with system and user messages to steer its behavior precisely. Temperature is set low for SQL generation and fixing to promote consistency, whereas for summarization a slightly higher creativity could be allowed (though we often keep it low or medium to ensure it sticks to facts).

Note: GPT-4 API does not use any data for training by default, and we are only sending de-identified or aggregated information to it (no raw patient identifiers), which is important for patient privacy.

### Streamlit (Frontend)
The user interface is built with Streamlit, which is a Python framework for creating web apps, especially data science demos, with minimal effort.

We created a simple Streamlit app (app.py) that contains a text box for the user's question and handles displaying the output. When the user hits "Submit", it calls our sqlrag_pipeline function and then shows a spinner while the pipeline runs. The results are then nicely formatted: the answer is shown in Markdown (which Streamlit renders), the SQL query can be viewed in an expandable code box (with syntax highlighting), and the raw data (Pandas DataFrame) is also available in an expandable section.

Streamlit was chosen for its ease of use – it allowed us to focus on the pipeline logic and quickly get a working UI without dealing with HTML/JS. It's great for demo and internal use purposes. (In a production setting, one might expose this via a Flask/FASTAPI API or integrate into a larger app, but Streamlit suffices for our goals.)

### Python (backend logic)
The glue of everything is Python. We organized the code in a modular way:
- `embedding.py` handles all vector operations (embedding generation and similarity retrieval).
- `llm.py` handles interactions with the OpenAI API and prompt formatting for all LLM tasks (SQL generation, fixing, summarization, classification).
- `db_utils.py` provides database connectivity and query execution, plus validation helpers.
- `pipeline.py` orchestrates the end-to-end flow, calling the above components in sequence and handling error cases.

This modular design makes it easy to maintain or extend. For example, one could swap out OpenAI for another LLM provider by editing llm.py, or replace the database with another source by adjusting db_utils.py, without affecting the other parts too much. We also isolate configuration (like DB connection params, API keys, thresholds) in config.py for clarity.

### MIMIC-IV dataset (v2.2, hosp module)
While not a "technology", it's the foundational data source for this project. MIMIC-IV is a large, freely accessible (with credentials) database of de-identified health records from intensive and general hospital care at the Beth Israel Deaconess Medical Center.

Our subset focuses on the hospital stay data ("hosp" module) – which includes patients, admissions, diagnosis codes, procedures, laboratory tests, medications, etc. There are on the order of 300k patients and 400k hospital admissions in this module.

We loaded these data into Postgres tables. The metadata table we constructed (named mimic_table_metadata) is a custom addition where we wrote descriptions and extra info for each table. That table, plus the pgVector extension, is what enables the retrieval step.

The actual patient data remains in the original tables, untouched except possibly for adding some indexes to speed up joins if necessary. We emphasize that privacy is preserved through de-identification (all patient identifiers are random IDs, dates are shifted, etc.) and our pipeline never tries to identify individuals or output raw personal data – it's geared toward aggregate analysis and general insights.

## Dataset Details and Ethical Considerations

MIMIC-IV (Medical Information Mart for Intensive Care IV) is a large, publicly available critical care database developed by MIT. It contains data from patients admitted to a large academic medical center from 2008 to 2019. MIMIC-IV is split into modules, and the hospital module ("hosp") includes data related to patients' hospital stays outside the ICU, as well as dictionary tables for codes:

- **Patients**: Demographics of patients (each has a unique subject_id, with attributes like gender, anchor age at admission, etc.). In our subset ~299k patients are present.
- **Admissions**: Records of each hospital admission (hadm_id), including admission/discharge times, type of admission, discharge disposition (home, died, etc.), and other administrative info.
- **Diagnoses (ICD codes)**: The diagnoses_icd table listing diagnoses for each admission (with references to ICD-10 or ICD-9 codes). There are also dictionary tables d_icd_diagnoses that map ICD codes to their long titles (descriptions of the diagnosis).
- **Procedures (ICD codes)**: Similar to diagnoses, procedures_icd and d_icd_procedures (though in our subset we might or might not include procedures – our focus was more on diagnoses, labs, and meds).
- **DRG Codes**: The drgcodes table which has Diagnostic Related Group codes for admissions (a way hospitals categorize admission costs/severity).
- **Laboratory tests**: The labevents table, which is quite large (millions of rows), containing results of lab tests (blood tests, etc.) for patients. Each lab event links to d_labitems which describes the lab test (e.g., itemid for "Glucose" test).
- **Medications**: The emar table (Electronic Medication Administration Record) which logs medications given to patients during their hospital stay (with timestamps, medication names, etc.). There may also be a pharmacy or prescriptions table in MIMIC-IV; in our subset we included emar as a representative source of medication data.
- **HCPCS codes**: A table d_hcpcs for procedure/service codes (mostly outpatient billing codes).
- (Other tables from MIMIC-IV hosp like services, transfers, etc., might be included in full MIMIC but our subset may skip some if not needed for demonstration.)

All data in MIMIC-IV are de-identified to protect patient privacy – names, exact dates, and other direct identifiers have been removed or altered. Nonetheless, the data is sensitive healthcare information, and we must handle it ethically:

- We ensure that our system does not reconstruct or expose any individual's identity or personal health information. The typical questions answered are aggregate or population-level (counts, averages, common occurrences) rather than "What happened to patient X?". If a user asked something like "Show me all data for patient 12345", our query classifier would flag that as likely private_data or at least not a supported query (since it's not a useful analytic question and borders on identifying a person).
- The query classification step explicitly looks for queries that might violate privacy. For example, "Who is patient 10000001?" or "Give me the names of patients with HIV" would be classified as disallowed. The system would refuse to provide an answer, even though MIMIC doesn't contain real names (the question itself implies trying to identify someone).
- We do not allow free-text generation about individuals. All outputs are either aggregated results or based on patterns in data that apply to groups.

One ethical point is using the OpenAI API with medical data. We are only sending schema descriptions and numerical results to OpenAI, not raw notes or any potentially identifying string. OpenAI states that API data is not used for training and is kept confidential. However, for a real clinical deployment, one might consider self-hosted models to avoid any external data transfer. Our use case is research-oriented on an already public dataset, so the risk is minimal and was deemed acceptable under PhysioNet's data use agreement.

### Limitations of the Data and Answers
The answers our system gives are only as good as the data in MIMIC-IV and the queries we run. MIMIC-IV is a single-center dataset; it may not be representative of all hospitals. Also, the data is retrospective from 2008-2019 – so any medical developments after 2019 (new treatments, COVID-19, etc.) are not in the data.

We attempt to flag queries that clearly ask for beyond-scope information (like asking about COVID incidence, which would be "future_data" relative to MIMIC's timeframe). But subtler issues remain: for example, if someone asks "What is the survival rate of patients on ventilators?", our system might produce an answer based on MIMIC-IV data, but that rate might not generalize to other hospitals or current practice. Users should be aware of these limitations.

Additionally, while GPT-4 is very advanced, it is not infallible:

- It might sometimes misinterpret a question's intent. We try to mitigate this with good prompting and the query classification step. If it does misinterpret, the SQL (and thus answer) might be answering a slightly different question than intended. Users should carefully word their questions and possibly rephrase if the result seems off.
- The summarization might occasionally introduce a slight hallucination or incorrect inference if the results are ambiguous. We provide the raw results for transparency so the user can verify what the data says. For instance, if the data shows a correlation, GPT-4 might use language like "suggests that ..." which is fine, but it should not claim causation or external facts not in data. In testing, GPT-4 tended to be quite factual when the result data was given explicitly.

Performance-wise, querying a large table like labevents can be slow. We have added an index and use appropriate filters in SQL, but some complex queries could approach the timeout. This is a technical limitation; in a production scenario, further optimizations or summarizing heavy tables in advance might be needed. Our pipeline warns or truncates results beyond a certain size to handle the volume.

In summary, ethical use of this system means using it for legitimate research or exploratory analysis, not attempting to identify individuals or feed it questions that it's not designed to answer. And any insights drawn should be validated and not taken as medical advice (the system is not a doctor or a clinical decision support tool; it's an analysis aid on a historical dataset).
