"""
Northbridge Bank - Credit Risk Natural-Language Query Engine (Streamlit app)

Converted from: Learner_Notebook_Project_3_Credit_Risk_Query_Engine.ipynb

Pipeline:
  classify_intent -> (verified template | generate_query) -> validate_query
  -> (retry_generation once if generated track fails) -> escalate OR execute_query
  -> generate_response

Read-only SQLite access, full SQL / data / confidence shown with every answer,
and an audit trail (in-session + append-only JSONL file).
"""

import os
import re
import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Query Engine | Northbridge Bank",
    page_icon="🏦",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "credit_risk_portfolio.db")
TEST_CSV_PATH = os.path.join(BASE_DIR, "test_queries.csv")
AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", os.path.join(BASE_DIR, "audit_log.jsonl"))
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


# ----------------------------------------------------------------------------
# Credentials / LLM setup
# (Notebook: loaded from config.json -> environment variables)
# Priority: Streamlit secrets -> environment variables -> config.json
# ----------------------------------------------------------------------------
def load_credentials():
    api_key, api_base = None, None

    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
        api_base = st.secrets.get("OPENAI_API_BASE", None)
    except Exception:
        pass

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    api_base = api_base or os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")

    config_path = os.path.join(BASE_DIR, "config.json")
    if (not api_key or not api_base) and os.path.exists(config_path):
        with open(config_path, "r") as file:
            config = json.load(file)
            api_key = api_key or config.get("OPENAI_API_KEY")
            api_base = api_base or config.get("OPENAI_API_BASE")

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_BASE_URL"] = api_base
    return api_key, api_base


@st.cache_resource(show_spinner=False)
def get_llms():
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0
    )

    evaluator_llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0
    )
    return llm, evaluator_llm


API_KEY, API_BASE = load_credentials()
if not API_KEY:
    st.error(
        "OPENAI_API_KEY is not configured. Add it to `.streamlit/secrets.toml` "
        "(or Streamlit Cloud → App settings → Secrets), set it as an environment "
        "variable, or provide a `config.json`."
    )
    st.stop()

llm, evaluator_llm = get_llms()


# ----------------------------------------------------------------------------
# Database loading (read-only connection)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_connection():
    # Read-only connection: URI mode with ?mode=ro prevents any write operation.
    # check_same_thread=False because Streamlit may run reruns on different threads.
    return sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True,
        check_same_thread=False
    )


if not os.path.exists(DB_PATH):
    st.error(f"Database file not found: `{DB_PATH}`. Place `credit_risk_portfolio.db` next to app.py.")
    st.stop()

conn = get_connection()


@st.cache_data(show_spinner=False)
def load_ground_truth():
    if os.path.exists(TEST_CSV_PATH):
        return pd.read_csv(TEST_CSV_PATH)
    return None


ground_truth = load_ground_truth()

tables = [
    "sector_master",
    "loan_master",
    "borrower_rating",
    "provisioning"
]


# ----------------------------------------------------------------------------
# Database schema provided to the LLM in prompts
# ----------------------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# ----------------------------------------------------------------------------
# Verified Query Template Library
# ----------------------------------------------------------------------------

# VQ1: Sector-wise Outstanding and NPA Breakdown
sql_1 = """
SELECT
    sm.sector_name,
    ROUND(SUM(lm.total_outstanding) / 1000000.0, 2) AS total_outstanding_mn,
    ROUND(
        SUM(
            CASE
                WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                THEN lm.total_outstanding
                ELSE 0
            END
        ) / 1000000.0,
        2
    ) AS npa_outstanding_mn
FROM loan_master lm
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
GROUP BY sm.sector_name
ORDER BY total_outstanding_mn DESC
"""

# VQ2: Portfolio Outstanding by Loan Category
sql_2 = """SELECT
    loan_category,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_mn DESC
"""

# VQ3: IFRS 9 Stage-wise ECL Summary
sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(*) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_mn,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage
"""

# VQ4: Provision Coverage Ratio by Sector
sql_4 = """
SELECT
    sm.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_ratio
FROM provisioning p
JOIN loan_master lm
    ON p.loan_account_number = lm.loan_account_number
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY sm.sector_name
ORDER BY avg_provision_coverage_ratio DESC
"""

# VQ5: Top 10 Loan Exposures
sql_5 = """
SELECT
    lm.borrower_name,
    sm.sector_name,
    ROUND(lm.total_outstanding / 1000000.0, 2) AS total_outstanding_mn,
    lm.asset_classification
FROM loan_master lm
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
ORDER BY lm.total_outstanding DESC
LIMIT 10
"""

# VQ6: Top 5 Business Group Exposures
sql_6 = """
SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master
WHERE group_name IS NOT NULL
  AND TRIM(group_name) <> ''
GROUP BY group_name
ORDER BY total_outstanding_mn DESC
LIMIT 5
"""

# VQ7: All Overdue Loan Accounts
sql_7 = """
SELECT
    lm.loan_account_number,
    lm.borrower_name,
    sm.sector_name,
    ROUND(lm.total_outstanding / 1000000.0, 2) AS total_outstanding_mn,
    lm.days_past_due,
    lm.asset_classification
FROM loan_master lm
JOIN sector_master sm
    ON lm.sector_code = sm.sector_code
WHERE lm.days_past_due > 0
ORDER BY lm.days_past_due DESC
"""

# VQ8: DPD Bucket Distribution
sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master
GROUP BY
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END
ORDER BY
    CASE
        WHEN dpd_bucket = '0 (Current)' THEN 1
        WHEN dpd_bucket = '1-30' THEN 2
        WHEN dpd_bucket = '31-60' THEN 3
        WHEN dpd_bucket = '61-90' THEN 4
        WHEN dpd_bucket = '90+' THEN 5
    END
"""

# VQ9: Latest Rating Downgrades
sql_9 = """
SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""

# VQ10: ECL Trend Across Reporting Quarters
sql_10 = """
SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date
"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },


    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },


    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },


    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },


    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },


    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },


    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },


    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },


    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },


    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}


# ----------------------------------------------------------------------------
# Tool: Intent Classification
# ----------------------------------------------------------------------------
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""

You are the routing component of an internal commercial lending credit-risk
query engine.

Compare the user's question against the verified query templates.

Choose "verified" when one template answers the same underlying analytical
request. Match the business metric, aggregation level, date logic, ranking,
and purpose, not merely similar keywords.

A verified template may return a complete result set even if the user asks
about one specific sector, category, stage, borrower, or DPD bucket. For
example, a question about NPA exposure in one sector should use the
sector-level NPA template because the response-generation step can select
the relevant sector from the complete result.

Choose "generated" when:
- no verified template covers the requested metric;
- the requested analytical grain is materially different;
- the requested date differs from the template;
- the question combines calculations from multiple templates;
- or the question is ambiguous.

Rules:
- Use only VQ1 through VQ10.
- Never invent a query ID.
- If route is "generated", query_id must be null.
- If route is "verified", query_id must contain the matching template ID.

USER QUESTION:
{user_question}

VERIFIED QUERY TEMPLATES:
{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ----------------------------------------------------------------------------
# Tool: Query Generation
# ----------------------------------------------------------------------------
def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""

You are a credit-risk analyst writing SQLite SQL.

Write one read-only SQL query that answers the user question.

Rules:
- Return SQL only, with no explanation or markdown.
- Start with SELECT or WITH.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or PRAGMA.
- Use only tables and columns in the schema.
- Apply every requested filter, grouping, date, sort, and limit.
- Outstanding exposure means total_outstanding unless otherwise specified.
- NPA or impaired means asset_classification IN
  ('Substandard', 'Doubtful', 'Loss').
- Latest provisioning date is '2025-09-30'.
- Latest rating date is '2025-09-30'.
- Use table aliases for columns when joining tables.
- Do not join tables unless required.
- If detailed records and an overall statistic are requested together,
  preserve every detailed row and calculate the overall statistic across
  the full filtered population.
- Use standard SQLite syntax only.

User question:
{user_question}

Database schema:
{schema_context}

Return only SQL.
"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# ----------------------------------------------------------------------------
# Tool: Query Validation (five checks)
# ----------------------------------------------------------------------------
def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }



    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result



    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select','from','where','and','or','group','by','order','having','limit','join','on','as','case',
                    'when','then','else','end','sum','count','avg','min','max','round','desc','asc','left','right',
                    'inner','outer','distinct','null','is','not','in','like','with','union','all','between','coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]



    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result



    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )


    relevance_prompt = f"""

You are validating SQLite SQL for a credit-risk query engine.

Decide whether the SQL correctly answers the user question.

Check:
- correct tables and joins;
- correct columns and metrics;
- correct filters and dates;
- correct grouping, sorting, and limits;
- correct NPA definition;
- correct result detail level;
- no duplicate counting.

A question may request detailed rows plus an overall statistic. In that case,
the overall statistic should cover the full filtered population and may be
repeated on each detailed row. It does not need to describe only the current
row.

Track context:
{track_context}

User question:
{user_question}

Candidate SQL:
{candidate_sql}

Return ONLY valid JSON:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result



    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# ----------------------------------------------------------------------------
# Tool: Retry Generation
# ----------------------------------------------------------------------------
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
Correct the failed SQLite query using the validation error.

Rules:
- Return one SQL query only.
- Start with SELECT or WITH.
- Use only tables and columns in the schema.
- Preserve the user's requested detail level, filters, calculations, dates,
  sorting, and limits.
- Correct the exact validation problem.
- NPA or impaired means asset_classification IN
  ('Substandard', 'Doubtful', 'Loss').
- If detailed rows and an overall statistic are requested together, preserve
  every detailed row and calculate the statistic across the full filtered
  population.
- Use standard SQLite syntax only.
- Do not include markdown or explanations.

User question:
{user_question}

Failed SQL:
{failed_sql}

Validation error:
{error_message}

Database schema:
{schema_context}

Return only the corrected SQL.
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# ----------------------------------------------------------------------------
# Tool: Query Execution (deterministic, read-only)
# ----------------------------------------------------------------------------
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# ----------------------------------------------------------------------------
# Tool: Response Generation
# ----------------------------------------------------------------------------
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""
Answer the user question using only the query result.

Rules:
- Do not invent or change values.
- Columns ending in "_mn" are USD millions.
- Keep the response concise and business-focused.
- Focus only on the requested information.
- If the data contains detailed rows and repeated overall totals, summarize
  the overall totals once.
- If the result is empty, say no matching records were found.

User question:
{user_question}

Route:
{route}

Query ID:
{query_id}

Query result:
{dataframe.to_string(index=False)}

Return only the final answer.
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# ----------------------------------------------------------------------------
# Pipeline orchestration
# (Notebook version printed intermediate stages; here they are collected in a
#  `trace` list that the UI renders. Logic is unchanged.)
# ----------------------------------------------------------------------------
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True, trace=None):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, records intermediate pipeline stages.
    - trace (list, optional): List that receives the intermediate stage messages.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    def say(msg):
        if verbose:
            print(msg)
            if trace is not None:
                trace.append(msg)

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    say(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    say(f"    Reason: {log['match_reason']}")


    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    say(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")


    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    say(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        say(f"    Failed check: {gate.get('failed_check')}")
        say(f"    Details: {gate.get('details')}")


    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        say(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        say(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            say(f"    Retry failed check: {gate.get('failed_check')}")
            say(f"    Retry details: {gate.get('details')}")


    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        say(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}


    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    say(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        say(f"    Warnings: {exec_result['warnings']}")


    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative


    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    say(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# ----------------------------------------------------------------------------
# Audit trail helpers
# ----------------------------------------------------------------------------
def build_audit_record(result, trace):
    log = result['log']
    return {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'user_question': log['user_question'],
        'route': log['route'],
        'query_id': log['query_id'],
        'match_reason': log['match_reason'],
        'candidate_sql': log['candidate_sql'],
        'executed_sql': log['executed_sql'],
        'gate_result': log['gate_result'],
        'retry_used': log['retry_used'],
        'escalated': log['escalated'],
        'row_count': log['row_count'],
        'confidence': log['confidence'],
        'narrative': log['narrative'],
        'pipeline_trace': trace,
        'model': MODEL_NAME,
    }


def persist_audit_record(record):
    """Append-only JSONL audit log. Never blocks the app if the file system is read-only."""
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return True
    except Exception:
        return False


def format_confidence(conf):
    if isinstance(conf, (int, float)):
        return f"{conf:.2f}"
    return str(conf) if conf is not None else "N/A"


# ----------------------------------------------------------------------------
# UI rendering helpers
# ----------------------------------------------------------------------------
def render_result(result, trace=None):
    """Render one pipeline result: answer, confidence, SQL, raw data, validation, audit."""
    log = result['log']
    gate = log.get('gate_result') or {}

    if log['escalated']:
        st.error("⚠️ Escalated to a human analyst — the engine could not reliably resolve this question.")
        st.write(log['narrative'])
    else:
        st.subheader("Answer")
        st.write(log['narrative'])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Route", str(log['route']).title())
    c2.metric("Query ID", log['query_id'] if log['query_id'] else "—")
    c3.metric("Confidence", format_confidence(log['confidence']))
    c4.metric("Rows returned", log['row_count'] if log['row_count'] is not None else "—")
    c5.metric("Retry used", "Yes" if log['retry_used'] else "No")

    st.caption(f"**Routing reason:** {log['match_reason']}")

    sql_to_show = log['executed_sql'] or log['candidate_sql']
    st.markdown("**SQL query used**" if log['executed_sql'] else "**Candidate SQL (not executed)**")
    st.code(sql_to_show or "", language="sql")

    if result['dataframe'] is not None:
        st.markdown("**Raw data returned**")
        st.dataframe(result['dataframe'], use_container_width=True)
        st.download_button(
            "Download result as CSV",
            data=result['dataframe'].to_csv(index=False).encode("utf-8"),
            file_name=f"query_result_{(log['query_id'] or 'generated')}.csv",
            mime="text/csv",
            key=f"dl_{abs(hash((log['user_question'], sql_to_show)))}_{id(result)}",
        )

    with st.expander("Validation gate details"):
        st.json(gate)

    if trace:
        with st.expander("Pipeline trace"):
            st.code("\n".join(trace), language="text")


def run_and_record(question):
    """Run the pipeline, record the audit trail, return (result, trace)."""
    trace = []
    result = run_pipeline(
        question,
        conn,
        verified_query_library,
        database_schema,
        verbose=True,
        trace=trace,
    )
    record = build_audit_record(result, trace)
    st.session_state.setdefault("history", []).append(record)
    persist_audit_record(record)
    return result, trace


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🏦 Northbridge Bank")
    st.caption("Credit Risk Analytics · Natural-Language Query Engine (PoC)")
    st.markdown(
        """
**How it works**
1. Your question is routed to a **verified SQL template** or a **generated** query.
2. SQL passes a **5-check validation gate** (read-only, schema, EXPLAIN, LLM relevance, template integrity).
3. Generated SQL gets **one retry**, then **escalates** to a human analyst.
4. Every answer shows the **SQL, raw data and confidence score**, and is logged for audit.
        """
    )
    st.divider()
    st.markdown("**Scope:** commercial lending portfolio · read-only · SELECT only")
    st.markdown(f"**Model:** `{MODEL_NAME}`")
    st.markdown(f"**Database:** `{os.path.basename(DB_PATH)}` (read-only)")
    st.markdown("**Latest reporting / rating date:** 2025-09-30")
    st.markdown("**NPA:** Substandard, Doubtful, Loss")


# ----------------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------------
st.title("Credit Risk Query Engine")
st.caption("Ask routine commercial-lending portfolio questions in plain English and get verified, auditable answers.")

tab_ask, tab_library, tab_eval, tab_audit = st.tabs(
    ["💬 Ask a Question", "📚 Verified Library & Schema", "🧪 Evaluation", "🧾 Audit Trail"]
)

# ---------------------------- Tab 1: Ask ------------------------------------
with tab_ask:
    example_questions = ["Show the distribution of loans across DPD buckets"]
    if ground_truth is not None and "User Query" in ground_truth.columns:
        example_questions = list(ground_truth["User Query"].dropna()) + example_questions

    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""

    def _load_example():
        choice = st.session_state.get("example_choice")
        if choice and choice != "— select an example —":
            st.session_state["question_input"] = choice

    st.selectbox(
        "Load an example question",
        ["— select an example —"] + example_questions,
        key="example_choice",
        on_change=_load_example,
    )

    question = st.text_area(
        "Your question",
        key="question_input",
        height=100,
        placeholder="e.g. What is the total NPA exposure in the Real Estate sector?",
    )

    run_clicked = st.button("Run query", type="primary", disabled=not question.strip())

    if run_clicked and question.strip():
        with st.spinner("Running pipeline: classify → build SQL → validate → execute → respond..."):
            try:
                result, trace = run_and_record(question.strip())
                st.session_state["last_result"] = (result, trace)
            except Exception as e:
                st.session_state.pop("last_result", None)
                st.error(f"The pipeline failed unexpectedly: {e}")
                st.info("Please rephrase the question or escalate to a human analyst.")

    if "last_result" in st.session_state:
        res, tr = st.session_state["last_result"]
        st.divider()
        st.markdown(f"**Question:** {res['log']['user_question']}")
        render_result(res, tr)

# ----------------------- Tab 2: Library & Schema ----------------------------
with tab_library:
    st.subheader("Verified Query Template Library")
    st.caption(f"{len(verified_query_library)} pre-approved, version-controlled SQL templates.")
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry['description']}"):
            st.code(entry['sql'].strip(), language="sql")

    st.subheader("Database Schema (provided to the LLM)")
    with st.expander("View schema"):
        st.code(database_schema.strip(), language="text")

    st.subheader("Table Preview")
    table_choice = st.selectbox("Table", tables)
    st.dataframe(
        pd.read_sql_query(f"SELECT * FROM {table_choice} LIMIT 5", conn),
        use_container_width=True,
    )

# --------------------------- Tab 3: Evaluation ------------------------------
with tab_eval:
    st.subheader("Evaluation Against Ground Truth")
    if ground_truth is None:
        st.warning("`test_queries.csv` not found next to app.py — evaluation is unavailable.")
    else:
        st.caption(f"{ground_truth.shape[0]} test cases loaded from test_queries.csv")
        st.dataframe(ground_truth, use_container_width=True)

        if st.button("Run all test cases"):
            test_results, test_traces = [], []
            progress = st.progress(0.0, text="Running test cases...")
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                try:
                    tr_result, tr_trace = run_and_record(gt['User Query'])
                except Exception as e:
                    tr_result = {
                        'log': {'user_question': gt['User Query'], 'route': 'error', 'query_id': None,
                                'match_reason': str(e), 'candidate_sql': None, 'gate_result': {},
                                'retry_used': False, 'escalated': True, 'executed_sql': None,
                                'row_count': None, 'confidence': None, 'narrative': f"Error: {e}"},
                        'dataframe': None,
                        'route': 'error', 'query_id': None, 'confidence': None, 'row_count': None,
                    }
                    tr_trace = [f"Error: {e}"]
                test_results.append(tr_result)
                test_traces.append(tr_trace)
                progress.progress((i + 1) / len(ground_truth), text=f"Completed {i + 1}/{len(ground_truth)}")
            progress.empty()
            st.session_state["eval_results"] = (test_results, test_traces)

        if "eval_results" in st.session_state:
            test_results, test_traces = st.session_state["eval_results"]

            evaluation_rows = []
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                if i >= len(test_results):
                    break
                tr = test_results[i]

                evaluation_rows.append({
                    'Test Case': gt['Test Case'],
                    'Expected Route': gt['Expected Route'],
                    'Actual Route': tr['route'],
                    'Route Match': tr['route'] == gt['Expected Route'],
                    'Expected Query ID': gt['Expected Query ID'],
                    'Actual Query ID': tr['query_id'],
                    'Query ID Match': (
                        pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
                    ) or tr['query_id'] == gt['Expected Query ID'],
                    'Confidence': tr['confidence'],
                    'Rows Returned': tr['row_count']
                })

            evaluation_df = pd.DataFrame(evaluation_rows)

            path_accuracy = evaluation_df['Route Match'].mean() * 100

            verified = evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
            query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100

            average_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce').mean()

            m1, m2, m3 = st.columns(3)
            m1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
            m2.metric("Selected Query Accuracy", "N/A" if pd.isna(query_accuracy) else f"{query_accuracy:.1f}%")
            m3.metric("Average Confidence Score", "N/A" if pd.isna(average_confidence) else f"{average_confidence:.2f}")

            st.dataframe(evaluation_df, use_container_width=True)
            st.dataframe(evaluation_df[['Test Case', 'Confidence']], use_container_width=True)

            st.markdown("**Per-test-case output**")
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                if i >= len(test_results):
                    break
                with st.expander(f"{gt['Test Case']}: {gt['User Query']}"):
                    if 'Expected Answer' in gt:
                        st.markdown(f"**Expected answer:** {gt['Expected Answer']}")
                    render_result(test_results[i], test_traces[i])

# ---------------------------- Tab 4: Audit ----------------------------------
with tab_audit:
    st.subheader("Audit Trail")
    st.caption(
        "Every system-generated output is recorded with the question, route, SQL, validation result, "
        "confidence and narrative. Records are kept for this session and appended to "
        f"`{os.path.basename(AUDIT_LOG_PATH)}`."
    )

    history = st.session_state.get("history", [])
    if not history:
        st.info("No queries have been run in this session yet.")
    else:
        summary_df = pd.DataFrame([{
            'Time (UTC)': h['timestamp_utc'],
            'Question': h['user_question'],
            'Route': h['route'],
            'Query ID': h['query_id'],
            'Confidence': h['confidence'],
            'Retry': h['retry_used'],
            'Escalated': h['escalated'],
            'Rows': h['row_count'],
        } for h in history])
        st.dataframe(summary_df, use_container_width=True)

        st.download_button(
            "Download full audit log (JSON)",
            data=json.dumps(history, indent=2, default=str).encode("utf-8"),
            file_name="audit_log_session.json",
            mime="application/json",
        )

        with st.expander("Full records"):
            for h in reversed(history):
                st.markdown(f"**{h['timestamp_utc']}** — {h['user_question']}")
                st.json(h)
