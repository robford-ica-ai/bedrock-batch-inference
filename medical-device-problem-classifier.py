import json
import os
import boto3
import pandas as pd
import numpy as np
import time
import random
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

def llm_wrapper(prompt: str, system_prompt: str,
    model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0",
    max_tokens: int = 4000,
    temperature: float = 0.1
) -> Dict[str, Any]:
    """
    A wrapper function for LLM API calls with improved error handling
    Args:
        prompt: The user prompt
        system_prompt: The system instructions
        model_id: The model identifier
        max_tokens: Maximum tokens in response
        temperature: Randomness parameter
    Returns:
        Dictionary with response text and token usage
    """
    bedrock = boto3.client('bedrock-runtime')
    # Calculate approximate input token count for logging
    approx_input_tokens = len(prompt.split()) + len(system_prompt.split())
    if approx_input_tokens > 30000:  # Rough threshold, adjust based on your model
        print(f"WARNING: Input may exceed token limits! (~{approx_input_tokens} words)")
    
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    try:
        print(f"Calling model: {model_id}")
        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        response_body = json.loads(response['body'].read())
        
        input_tokens = response_body.get('usage', {}).get('input_tokens', 0)
        output_tokens = response_body.get('usage', {}).get('output_tokens', 0)
        print(f"API call successful: {input_tokens} input tokens, {output_tokens} output tokens")
        if input_tokens == 0 and output_tokens == 0:
            print("WARNING: Zero tokens reported, but no error thrown!")

        response_text = response_body['content'][0]['text'].strip()

        return {
            'text': response_text,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"API ERROR: {error_msg}")
        # More specific error handling
        if "AccessDeniedException" in error_msg:
            print("Access denied - check credentials and permissions")
        elif "ValidationException" in error_msg and "input tokens" in error_msg.lower():
            print("Input token limit exceeded!")
        elif "ThrottlingException" in error_msg:
            print("API rate limit exceeded")
        elif "ServiceUnavailableException" in error_msg:
            print("Service unavailable - try again later")
        return {
            'text': '',
            'input_tokens': 0,
            'output_tokens': 0,
            'error': error_msg
        }

class MedicalTextClassifier:
    def __init__(
        self,
        model_id: str,
        problem_type: str,
        batch_size: int,
        max_workers: int,
        rate_limit_per_second: int,
        llm_function: Callable = llm_wrapper,
        problem_labels: List[str] = [],
    ):
        """
        Initialize the medical text classifier
        Args:
            model_id: The model identifier to use
            problem_type: Type of problem to classify ("product" or "patient")
            batch_size: Number of texts to process in a single API call
            max_workers: Maximum number of concurrent threads
            rate_limit_per_second: Maximum API calls per second
            llm_function: Function to call the LLM API
            problem_labels: List of labels for classification
        """
        self.llm_function = llm_function
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.rate_limit = rate_limit_per_second
        self.problem_type = problem_type
        
        # Rate limiting variables
        self.api_call_timestamps = []
        self.rate_limit_lock = Lock()
        
        # Convert problem labels to a Python list if it's a string
        if isinstance(problem_labels, str):
            self.problem_label_list = [x.strip() for x in problem_labels.split('|') if x.strip()]
        else:
            self.problem_label_list = problem_labels
        
        # Store the problem labels as a string for prompt creation
        self.problem_labels = '|'.join(self.problem_label_list) if isinstance(problem_labels, list) else problem_labels
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Pre-cache the system prompt
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create the system prompt with instructions for the specific problem type."""
        problem_list = self.problem_labels
        classification_type = "product" if self.problem_type == "product" else "patient"
        return f"""You are a medical text validator analyzing medical device incident texts and classifying {classification_type} issues for each one.
     
    You must use these {classification_type} issue labels:
     
    {problem_list}
     
    For each text, you must:
    1. Validate if there's enough information to assign labels
    2. Assign a primary {classification_type} label as follows:
       - If there's sufficient information, use the most appropriate label from the list above
       - If there's insufficient information but you can make a reasonable inference, still assign the most likely label
       - If there's truly insufficient information to make any determination, use "Insufficient Information" as the primary_label
    3. Assign up to 3 supplementary labels when applicable
    4. Provide justification for your decisions
     
    Output MUST be valid JSON array where each element corresponds to a text in the input and follows this format:
     
    [
      {{
        "text_id": "the ID number of the text",
        "enough_text": true or false,
        "{classification_type}_classification": {{
          "primary_label": "chosen {classification_type} label or 'Insufficient Information' if truly cannot determine",
          "supplementary_labels": ["additional {classification_type} labels, empty list if none apply"],
          "justification": "detailed justification for {classification_type} labels or why information is insufficient",
          "problem_text": "text excerpt used for classification"
        }}
      }}
    ]
     
    IMPORTANT: Primary_label should reflect your best assessment, even with limited information. Only use "Insufficient Information" when no reasonable determination can be made.
    """

    def _create_batch_prompt(self, batch_data: List[Dict[str, Any]]) -> str:
        """Create a prompt containing multiple texts to classify with length validation."""
        texts = []
        total_chars = 0
        for idx, item in enumerate(batch_data):
            if pd.isna(item['text']) or not str(item['text']).strip():
                continue
            text = str(item['text']).strip()
            # Truncate very long texts to prevent token limit issues
            if len(text) > 3000:  # Choose a reasonable character limit
                text = text[:3000] + "... [text truncated for length]"
            formatted_text = f"TEXT #{idx+1} (ID: {item['idx']}, UID: {item['uid']}):\n{text}\n"
            texts.append(formatted_text)
            total_chars += len(formatted_text)
        # Rough token estimation
        approx_tokens = total_chars / 4  # ~4 chars per token on average
        if approx_tokens > 30000:  # Adjust based on model limits
            print(f"WARNING: Batch may exceed token limit! (~{approx_tokens} estimated tokens)")
        return f"Analyze these medical device incidents for {self.problem_type} issues:\n\n" + "\n".join(texts)

    def _rate_limit_api_calls(self):
        """Implement rate limiting for API calls."""
        with self.rate_limit_lock:
            current_time = time.time()
            # Remove timestamps older than 1 second
            self.api_call_timestamps = [ts for ts in self.api_call_timestamps 
                                     if current_time - ts < 1.0]
            # If we've reached the rate limit, sleep
            if len(self.api_call_timestamps) >= self.rate_limit:
                oldest_timestamp = min(self.api_call_timestamps)
                sleep_time = 1.0 - (current_time - oldest_timestamp)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    current_time = time.time()  # Update current time after sleep
            # Add timestamp for current call
            self.api_call_timestamps.append(current_time)

    def _clean_json_response(self, text: str) -> str:
        """Clean the response text to ensure valid JSON."""
        # Try to extract the JSON array from the response
        try:
            # Find the first '[' and last ']'
            start_idx = text.find('[')
            end_idx = text.rfind(']')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_text = text[start_idx:end_idx+1]
                # Test if it's valid JSON
                json.loads(json_text)
                return json_text
            
            # If not found, check for a single JSON object
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_text = text[start_idx:end_idx+1]
                # Wrap single object in an array
                test_obj = json.loads(json_text)
                return f"[{json_text}]"
        except:
            pass
        # If extraction fails, return the original text
        return text

    def _invoke_model_for_batch(self, batch_data: List[Dict[str, Any]]) -> Dict:
        """Invoke the model with multiple texts in a single prompt using the LLM wrapper."""
        # Filter out empty texts
        valid_batch = [item for item in batch_data if not pd.isna(item['text']) and str(item['text']).strip()]
        if not valid_batch:
            return {'classifications': [], 'input_tokens': 0, 'output_tokens': 0}
        
        user_prompt = self._create_batch_prompt(valid_batch)
        system_prompt = self.system_prompt

        max_retries = 3
        base_delay = 1  # seconds
        for attempt in range(max_retries):
            try:
                # Rate limit
                self._rate_limit_api_calls()
                response = self.llm_function(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    model_id=self.model_id
                )
                input_tokens = response.get('input_tokens', 0)
                output_tokens = response.get('output_tokens', 0)
                # Update total token counts
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                
                response_text = response.get('text', '')
                # Clean and parse the response
                cleaned_response = self._clean_json_response(response_text)
                try:
                    classifications = json.loads(cleaned_response)
                    # Ensure it's a list even if only one text was processed
                    if not isinstance(classifications, list):
                        classifications = [classifications]
                    return {
                        'classifications': classifications,
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens
                    }
                except json.JSONDecodeError as e:
                    fallback_classifications = []
                    for item in valid_batch:
                        if self.problem_type == "product":
                            fallback_classifications.append({
                                "text_id": f"ID: {item['idx']}, UID: {item['uid']}",
                                "enough_text": False,
                                "product_classification": {
                                    "primary_label": "na",
                                    "supplementary_labels": [],
                                    "justification": "Unable to classify due to parsing error",
                                    "problem_text": ""
                                }
                            })
                        else:  # patient
                            fallback_classifications.append({
                                "text_id": f"ID: {item['idx']}, UID: {item['uid']}",
                                "enough_text": False,
                                "patient_classification": {
                                    "primary_label": "na",
                                    "supplementary_labels": [],
                                    "justification": "Unable to classify due to parsing error",
                                    "problem_text": ""
                                }
                            })
                    return {
                        'classifications': fallback_classifications,
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'error': f'JSON parsing error: {str(e)}',
                        'raw_response': response_text
                    }
            except Exception as e:
                error_str = str(e)
                if "ThrottlingException" in error_str and attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                    print(f"Throttling detected (attempt {attempt+1}/{max_retries}), retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                elif attempt < max_retries - 1 and any(err in error_str for err in ["ServiceUnavailableException", "InternalServerException", "ConnectionError", "Timeout"]):
                    # Also retry on transient errors
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Transient error detected: {error_str[:100]}... (attempt {attempt+1}/{max_retries}), retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    if "ThrottlingException" in error_str:
                        print(f"Max retries exceeded for throttling. Consider reducing concurrency or request rate.")
                    return {
                        'error': f'API call failed: {error_str}',
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'classifications': []
                    }

    def _process_batch(self, batch_data: List[Dict[str, Any]], batch_idx: int) -> Tuple[int, List[Dict]]:
        """Process a batch of texts and return results with the batch index."""
        batch_result = self._invoke_model_for_batch(batch_data)
        results = []
        classifications = batch_result.get('classifications', [])
        
        # Match classifications with original data
        for i, classification in enumerate(classifications):
            text_id = classification.get('text_id')
            original_idx = None
            uid = None
            # Convert text_id to string to avoid "int is not iterable" error
            if text_id is not None:
                text_id = str(text_id)
                if 'UID:' in text_id:
                    uid_part = text_id.split('UID:')[1].strip()
                    uid = uid_part.split(',')[0].strip() if ',' in uid_part else uid_part.strip()
                    uid = uid.rstrip(')') if uid.endswith(')') else uid
            if uid is None and i < len(batch_data):
                uid = batch_data[i]['uid']
            if text_id is not None and 'ID:' in text_id:
                try:
                    idx_part = text_id.split('ID:')[1].strip()
                    idx_str = idx_part.split(',')[0].strip() if ',' in idx_part else idx_part.strip()
                    original_idx = int(idx_str)
                except (ValueError, IndexError):
                    original_idx = batch_data[i]['idx'] if i < len(batch_data) else None
            else:
                original_idx = batch_data[i]['idx'] if i < len(batch_data) else None
                
            results.append({
                'row_index': original_idx,
                'uid': uid,
                'classification': classification,
                'input_tokens': batch_result.get('input_tokens', 0) // max(len(classifications), 1),
                'output_tokens': batch_result.get('output_tokens', 0) // max(len(classifications), 1)
            })
            
        if not classifications and 'error' in batch_result:
            for item in batch_data:
                results.append({
                    'row_index': item['idx'],
                    'uid': item['uid'],
                    'classification': None,
                    'error': batch_result.get('error'),
                    'input_tokens': 0,
                    'output_tokens': 0
                })
                
        return batch_idx, results

    def classify_texts_from_list(self, 
                               texts: List[str],
                               uids: List[str] = None,
                               show_progress: bool = True) -> pd.DataFrame:
        """
        Classify medical texts using parallel batch processing from a list.
        Args:
            texts: List of texts to classify
            uids: List of unique identifiers
            show_progress: Whether to show a progress bar
        Returns:
            DataFrame with classification results
        """
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        if uids is None:
            uids = [str(i) for i in range(len(texts))]
            
        data = [{'idx': i, 'text': text, 'uid': uid} 
                for i, (text, uid) in enumerate(zip(texts, uids))]
        
        batches = [data[i:i + self.batch_size] for i in range(0, len(data), self.batch_size)]
        
        all_results = []
        completed_batches = {}
        next_batch_to_process = 0
        
        print(f"Processing {len(texts)} texts in {len(batches)} batches")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_batch, batch, idx): idx
                for idx, batch in enumerate(batches)
            }
            
            for future in as_completed(future_to_batch):
                batch_idx, batch_results = future.result()
                completed_batches[batch_idx] = batch_results
                
                while next_batch_to_process in completed_batches:
                    all_results.extend(completed_batches[next_batch_to_process])
                    del completed_batches[next_batch_to_process]
                    next_batch_to_process += 1
                    print(f"Completed batch {next_batch_to_process-1}/{len(batches)}")
        
        # In case anything is left (should not normally happen)
        for idx in sorted(completed_batches.keys()):
            all_results.extend(completed_batches[idx])
            
        processed_results = []
        
        for result in all_results:
            classification = result.get('classification', {})
            uid = result.get('uid')
            row_index = result.get('row_index')
            error = result.get('error')
            
            if classification:
                result_dict = {
                    'row_index': row_index,
                    'uid': uid,
                    'mdr_text': texts[row_index] if row_index is not None and row_index < len(texts) else None,
                    'input_tokens': result['input_tokens'],
                    'output_tokens': result['output_tokens'],
                    'error': None
                }
                if self.problem_type == "product":
                    result_dict.update({
                        'ai_sufficient_info_annex_a_product': classification.get('enough_text'),
                        'ai_assigned_annex_a_product': classification.get('product_classification', {}).get('primary_label'),
                        'ai_supporting_labels_annex_a_product': classification.get('product_classification', {}).get('supplementary_labels'),
                        'ai_justification_annex_a_product': classification.get('product_classification', {}).get('justification'),
                        'ai_justification_from_text_annex_a_product': classification.get('product_classification', {}).get('problem_text')
                    })
                else:  # patient
                    result_dict.update({
                        'ai_sufficient_info_annex_e_patient': classification.get('enough_text'),
                        'ai_assigned_annex_e_patient': classification.get('patient_classification', {}).get('primary_label'),
                        'ai_supporting_labels_annex_e_patient': classification.get('patient_classification', {}).get('supplementary_labels'),
                        'ai_justification_annex_e_patient': classification.get('patient_classification', {}).get('justification'),
                        'ai_justification_from_text_annex_e_patient': classification.get('patient_classification', {}).get('problem_text')
                    })
                processed_results.append(result_dict)
            else:
                # Something went wrong or no classifications
                result_dict = {
                    'row_index': row_index,
                    'uid': uid,
                    'mdr_text': texts[row_index] if row_index is not None and row_index < len(texts) else None,
                    'input_tokens': result['input_tokens'],
                    'output_tokens': result['output_tokens'],
                    'error': error
                }
                if self.problem_type == "product":
                    result_dict.update({
                        'ai_sufficient_info_annex_a_product': False,
                        'ai_assigned_annex_a_product': None,
                        'ai_supporting_labels_annex_a_product': None,
                        'ai_justification_annex_a_product': None,
                        'ai_justification_from_text_annex_a_product': None
                    })
                else:  # patient
                    result_dict.update({
                        'ai_sufficient_info_annex_e_patient': False,
                        'ai_assigned_annex_e_patient': None,
                        'ai_supporting_labels_annex_e_patient': None,
                        'ai_justification_annex_e_patient': None,
                        'ai_justification_from_text_annex_e_patient': None
                    })
                processed_results.append(result_dict)
        
        print(f"Total processing complete:")
        print(f"- Processed {len(processed_results)} texts in {len(batches)} batches")
        print(f"- Total input tokens: {self.total_input_tokens}")
        print(f"- Total output tokens: {self.total_output_tokens}")
        
        return pd.DataFrame(processed_results)
    
    def classify_medical_texts(self,
                              df: pd.DataFrame, 
                              text_col: str,
                              uid_col: str = None,
                              show_progress: bool = True) -> pd.DataFrame:
        """
        Classify medical texts from a DataFrame
        Args:
            df: DataFrame containing texts to classify
            text_col: Column name containing the text to analyze
            uid_col: Column name containing unique identifiers
            show_progress: Whether to show a progress bar
        Returns:
            DataFrame with classification results
        """
        texts = df[text_col].tolist()
        
        if uid_col:
            uids = df[uid_col].astype(str).tolist()
        else:
            uids = df.index.astype(str).tolist()
            
        results_df = self.classify_texts_from_list(
            texts=texts,
            uids=uids,
            show_progress=show_progress
        )
        
        return results_df

def process_result(result_df, problem_type):
    """
    Processes the final result for S3/Athena compatibility
    Args:
        result_df: DataFrame with classification results
        problem_type: Type of problem ("product" or "patient")
    Returns:
        Processed DataFrame ready for storage
    """
    try:
        processed_df = result_df.copy()

        # Process by problem type
        if problem_type == "product":
            # Strip and clean primary label
            processed_df['ai_assigned_annex_a_product'] = processed_df['ai_assigned_annex_a_product'].str.strip()
            
            # Drop irrelevant columns
            if 'row_index' in processed_df.columns:
                processed_df.drop(columns=['row_index'], inplace=True, errors='ignore')
            if 'error' in processed_df.columns:    
                processed_df.drop(columns=['error'], inplace=True, errors='ignore')
                
            # Fill NaN values
            processed_df = processed_df.fillna('')

            # Convert list columns to JSON strings
            processed_df['ai_supporting_labels_annex_a_product'] = (
                processed_df['ai_supporting_labels_annex_a_product']
                .apply(lambda x: json.dumps(x) if isinstance(x, list) else str(x))
            )

            processed_df['ai_assigned_annex_a_product'] = (
                processed_df['ai_assigned_annex_a_product']
                .apply(lambda x: json.dumps(x) if isinstance(x, list) else str(x))
            )

            processed_df['ai_justification_from_text_annex_a_product'] = (
                processed_df['ai_justification_from_text_annex_a_product']
                .apply(lambda x: json.dumps(x) if isinstance(x, list) else str(x))
            )

            processed_df['ai_justification_annex_a_product'] = (
                processed_df['ai_justification_annex_a_product']
                .apply(lambda x: json.dumps(x) if isinstance(x, list) else str(x))
            )
            
        else:  # patient
            # Strip and clean primary label
            processed_df['ai_assigned_annex_e_patient'] = processed_df['ai_assigned_annex_e_patient'].str.strip()
            
            # Replace 'na' with standard term
            processed_df.loc[processed_df['ai_assigned_annex_e_patient']=='na', 'ai_assigned_annex_e_patient'] = 'No Clinical Signs, Symptoms or Conditions'
            
            # Drop irrelevant columns
            if 'row_index' in processed_df.columns:
                processed_df.drop(columns=['row_index'], inplace=True, errors='ignore')
            if 'error' in processed_df.columns:    
                processed_df.drop(columns=['error'], inplace=True, errors='ignore')
                
            # Fill NaN values
            processed_df = processed_df.fillna('')

            # Convert list columns to JSON strings
            processed_df['ai_supporting_labels_annex_e_patient'] = (
                processed_df['ai_supporting_labels_annex_e_patient']
                .apply(lambda x: json.dumps(x) if isinstance(x, list) else str(x))
            )
            
            processed_df['ai_assigned_annex_e_patient'] = (
                processed_df['ai_assigned_annex_e_patient']
                .apply(lambda x: json.dumps(x) if isinstance(x, list) else str(x))
            )
            
            processed_df['ai_justification_from_text_annex_e_patient'] = (
                processed_df['ai_justification_from_text_annex_e_patient']
                .apply(lambda x: json.dumps(x) if isinstance(x, list) else str(x))
            )
            
            processed_df['ai_justification_annex_e_patient'] = (
                processed_df['ai_justification_annex_e_patient']
                .apply(lambda x: json.dumps(x) if isinstance(x, list) else str(x))
            )
            
        return processed_df

    except Exception as e:
        print(f"Error processing results: {e}")
        return result_df

def get_problem_labels(df, problem_type):
    """
    Creates a list of IMDRF terms based on the problem type. 
    - 'A' list for product issues 
    - 'E' list for patient issues
    Args:
        df: DataFrame containing IMDRF codes and terms
        problem_type: Type of problem ("product" or "patient")
    Returns:
        List of problem labels
    """
    prefix = "A" if problem_type == "product" else "E"
    problem_labels = list(df[df['code'].str.startswith(prefix)]['term'])
    problem_labels = sorted(set(problem_labels))
    return problem_labels

def get_s3_data(bucket, key):
    """
    Get data from S3
    Args:
        bucket: S3 bucket name
        key: S3 object key
    Returns:
        Content of the S3 object
    """
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket, Key=key)
    return response['Body'].read().decode('utf-8')

def write_s3_data(bucket, key, data):
    """
    Write data to S3
    Args:
        bucket: S3 bucket name
        key: S3 object key
        data: Data to write
    """
    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    
# Main batch inference handler for AWS Bedrock Batch Jobs
def lambda_handler(event, context):
    """
    AWS Lambda handler for Bedrock Batch Inference
    
    Args:
        event: The event dict containing job details
        context: Lambda context
    
    Returns:
        Dictionary with job results summary
    """
    try:
        # Extract parameters from the event
        input_bucket = event.get('input_bucket')
        input_key = event.get('input_key')
        output_bucket = event.get('output_bucket')
        output_prefix = event.get('output_prefix')
        model_id = event.get('model_id', 'anthropic.claude-3-5-sonnet-20240620-v1:0')
        problem_type = event.get('problem_type', 'product')  # 'product' or 'patient'
        batch_size = int(event.get('batch_size', 1))
        max_workers = int(event.get('max_workers', 5))
        rate_limit = int(event.get('rate_limit', 5))
        imdrf_bucket = event.get('imdrf_bucket', input_bucket)
        imdrf_key = event.get('imdrf_key', 'reference/imdrf.csv')
        
        print(f"Starting batch inference job with parameters:")
        print(f"- Input: s3://{input_bucket}/{input_key}")
        print(f"- Output: s3://{output_bucket}/{output_prefix}")
        print(f"- Model: {model_id}")
        print(f"- Problem type: {problem_type}")
        
        # Load IMDRF data for labels
        imdrf_content = get_s3_data(imdrf_bucket, imdrf_key)
        imdrf_df = pd.read_csv(pd.StringIO(imdrf_content))
        problem_labels = get_problem_labels(imdrf_df, problem_type)
        problem_labels_str = "|".join(sorted(problem_labels))
        
        # Load input data
        input_content = get_s3_data(input_bucket, input_key)
        input_df = pd.read_csv(pd.StringIO(input_content))
        
        # Initialize the classifier
        classifier = MedicalTextClassifier(
            model_id=model_id,
            problem_type=problem_type,
            batch_size=batch_size,
            max_workers=max_workers,
            rate_limit_per_second=rate_limit,
            problem_labels=problem_labels_str
        )
        
        # Process in chunks to manage memory
        chunk_size = 1000  # Adjust based on your data size
        total_chunks = (len(input_df) + chunk_size - 1) // chunk_size
        results = []
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(input_df))
            chunk_df = input_df.iloc[start_idx:end_idx]
            
            chunk_result = process_chunk(
                chunk_df, 
                classifier, 
                problem_type, 
                output_bucket, 
                output_prefix, 
                chunk_idx
            )
            results.append(chunk_result)
        
        # Summarize results
        total_records = sum(r.get('records_processed', 0) for r in results if 'records_processed' in r)
        total_input_tokens = sum(r.get('input_tokens', 0) for r in results if 'input_tokens' in r)
        total_output_tokens = sum(r.get('output_tokens', 0) for r in results if 'output_tokens' in r)
        total_cost = sum(r.get('estimated_cost', 0) for r in results if 'estimated_cost' in r)
        errors = [r.get('error') for r in results if 'error' in r]
        
        summary = {
            'job_id': context.aws_request_id,
            'status': 'COMPLETED' if not errors else 'COMPLETED_WITH_ERRORS',
            'total_records': total_records,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_estimated_cost': total_cost,
            'errors': errors if errors else None,
            'output_location': f"s3://{output_bucket}/{output_prefix}/"
        }
        
        # Write summary to S3
        write_s3_data(
            output_bucket, 
            f"{output_prefix}/job_summary.json", 
            json.dumps(summary, indent=2)
        )
        
        return summary
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error in batch job: {error_msg}")
        return {
            'status': 'FAILED',
            'error': error_msg
        }

def process_chunk(df_chunk, classifier, problem_type, output_bucket, output_prefix, batch_idx):
    """
    Process a chunk of data and write results to S3
    Args:
        df_chunk: DataFrame chunk to process
        classifier: MedicalTextClassifier instance
        problem_type: Type of problem ("product" or "patient")
        output_bucket: S3 bucket for output
        output_prefix: S3 prefix for output
        batch_idx: Index of the current batch
    Returns:
        Summary of processing statistics
    """
    try:
        print(f"Processing chunk {batch_idx} with {len(df_chunk)} records")
        
        # Classify the texts
        classification_results = classifier.classify_medical_texts(
            df_chunk,
            "mdr_text",
            "id"
        )
        
        # Process results for storage
        processed_results = process_result(classification_results, problem_type)
        
        # Convert to string for storage
        processed_csv = processed_results.to_csv(index=False)
        
        # Write to S3
        output_key = f"{output_prefix}/{problem_type}_results_batch_{batch_idx}.csv"
        write_s3_data(output_bucket, output_key, processed_csv)
        
        # Calculate token usage and cost estimates
        total_input_tokens = processed_results['input_tokens'].astype(float).sum()
        total_output_tokens = processed_results['output_tokens'].astype(float).sum()
        
        # Approximate Claude pricing (may need adjustment based on your contract)
        input_cost = (total_input_tokens / 1000) * 0.0036  # $3.60 per million input tokens
        output_cost = (total_output_tokens / 1000) * 0.018  # $18.00 per million output tokens
        total_cost = input_cost + output_cost
        
        print(f"Chunk {batch_idx} processing complete:")
        print(f"- Records processed: {len(processed_results)}")
        print(f"- Input tokens: {total_input_tokens:,.0f} (${input_cost:.2f})")
        print(f"- Output tokens: {total_output_tokens:,.0f} (${output_cost:.2f})")
        print(f"- Total estimated cost: ${total_cost:.2f}")
        
        return {
            "batch_idx": batch_idx,
            "records_processed": len(processed_results),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "estimated_cost": total_cost,
            "output_key": output_key
        }
    except Exception as e:
        print(f"Error processing chunk {batch_idx}: {str(e)}")
        return {
            "batch_idx": batch_idx,
            "error": str(e)
        }