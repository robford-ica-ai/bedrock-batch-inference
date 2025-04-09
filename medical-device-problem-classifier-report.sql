# Example Athena queries for analyzing classification results from AWS Bedrock Batch Inference

# 1. Create external table for product classification results
CREATE_PRODUCT_TABLE = """
CREATE EXTERNAL TABLE IF NOT EXISTS product_classifications (
  uid STRING,
  mdr_text STRING,
  input_tokens DOUBLE,
  output_tokens DOUBLE,
  ai_sufficient_info_annex_a_product BOOLEAN,
  ai_assigned_annex_a_product STRING,
  ai_supporting_labels_annex_a_product STRING,
  ai_justification_annex_a_product STRING,
  ai_justification_from_text_annex_a_product STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar' = '\"',
  'escapeChar' = '\\'
)
STORED AS TEXTFILE
LOCATION 's3://medical-device-analysis/classifications/product/2025-04-09/'
TBLPROPERTIES ('skip.header.line.count'='1');
"""

# 2. Create external table for patient classification results
CREATE_PATIENT_TABLE = """
CREATE EXTERNAL TABLE IF NOT EXISTS patient_classifications (
  uid STRING,
  mdr_text STRING,
  input_tokens DOUBLE,
  output_tokens DOUBLE,
  ai_sufficient_info_annex_e_patient BOOLEAN,
  ai_assigned_annex_e_patient STRING,
  ai_supporting_labels_annex_e_patient STRING,
  ai_justification_annex_e_patient STRING,
  ai_justification_from_text_annex_e_patient STRING
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar' = '\"',
  'escapeChar' = '\\'
)
STORED AS TEXTFILE
LOCATION 's3://medical-device-analysis/classifications/patient/2025-04-09/'
TBLPROPERTIES ('skip.header.line.count'='1');
"""

# 3. Query to find the most common product issues
QUERY_COMMON_PRODUCT_ISSUES = """
SELECT 
  JSON_EXTRACT_SCALAR(ai_assigned_annex_a_product, '$') as product_issue,
  COUNT(*) as count
FROM product_classifications
WHERE ai_sufficient_info_annex_a_product = true
GROUP BY JSON_EXTRACT_SCALAR(ai_assigned_annex_a_product, '$')
ORDER BY count DESC
LIMIT 10;
"""

# 4. Query to find the most common patient issues
QUERY_COMMON_PATIENT_ISSUES = """
SELECT 
  JSON_EXTRACT_SCALAR(ai_assigned_annex_e_patient, '$') as patient_issue,
  COUNT(*) as count
FROM patient_classifications
WHERE ai_sufficient_info_annex_e_patient = true
GROUP BY JSON_EXTRACT_SCALAR(ai_assigned_annex_e_patient, '$')
ORDER BY count DESC
LIMIT 10;
"""

# 5. Query to join product and patient issues for the same incidents
QUERY_JOIN_PRODUCT_PATIENT = """
SELECT 
  p.uid,
  JSON_EXTRACT_SCALAR(p.ai_assigned_annex_a_product, '$') as product_issue,
  JSON_EXTRACT_SCALAR(pt.ai_assigned_annex_e_patient, '$') as patient_issue,
  p.mdr_text
FROM 
  product_classifications p
JOIN 
  patient_classifications pt ON p.uid = pt.uid
WHERE 
  p.ai_sufficient_info_annex_a_product = true
  AND pt.ai_sufficient_info_annex_e_patient = true
LIMIT 100;
"""

# 6. Query for incidents with insufficient information
QUERY_INSUFFICIENT_INFO = """
SELECT 
  uid,
  mdr_text,
  JSON_EXTRACT_SCALAR(ai_justification_annex_a_product, '$') as justification
FROM 
  product_classifications
WHERE 
  ai_sufficient_info_annex_a_product = false
LIMIT 50;
"""

# 7. Query to calculate token usage statistics
QUERY_TOKEN_USAGE = """
SELECT 
  'product' as classification_type,
  COUNT(*) as record_count,
  SUM(input_tokens) as total_input_tokens,
  SUM(output_tokens) as total_output_tokens,
  AVG(input_tokens) as avg_input_tokens_per_record,
  AVG(output_tokens) as avg_output_tokens_per_record,
  SUM(input_tokens) * 0.0000036 as input_cost_usd,
  SUM(output_tokens) * 0.000018 as output_cost_usd,
  (SUM(input_tokens) * 0.0000036) + (SUM(output_tokens) * 0.000018) as total_cost_usd
FROM 
  product_classifications

UNION ALL

SELECT 
  'patient' as classification_type,
  COUNT(*) as record_count,
  SUM(input_tokens) as total_input_tokens,
  SUM(output_tokens) as total_output_tokens,
  AVG(input_tokens) as avg_input_tokens_per_record,
  AVG(output_tokens) as avg_output_tokens_per_record,
  SUM(input_tokens) * 0.0000036 as input_cost_usd,
  SUM(output_tokens) * 0.000018 as output_cost_usd,
  (SUM(input_tokens) * 0.0000036) + (SUM(output_tokens) * 0.000018) as total_cost_usd
FROM 
  patient_classifications;
"""

# 8. Query to find co-occurring product and patient issues
QUERY_COOCCURRING_ISSUES = """
SELECT 
  JSON_EXTRACT_SCALAR(p.ai_assigned_annex_a_product, '$') as product_issue,
  JSON_EXTRACT_SCALAR(pt.ai_assigned_annex_e_patient, '$') as patient_issue,
  COUNT(*) as occurrence_count
FROM 
  product_classifications p
JOIN 
  patient_classifications pt ON p.uid = pt.uid
WHERE 
  p.ai_sufficient_info_annex_a_product = true
  AND pt.ai_sufficient_info_annex_e_patient = true
GROUP BY 
  JSON_EXTRACT_SCALAR(p.ai_assigned_annex_a_product, '$'),
  JSON_EXTRACT_SCALAR(pt.ai_assigned_annex_e_patient, '$')
HAVING 
  COUNT(*) > 5
ORDER BY 
  occurrence_count DESC
LIMIT 20;
"""