# Example job configurations for AWS Bedrock Batch Inference

# 1. Sample JSON configuration for product issue classification
PRODUCT_JOB_CONFIG = {
    "input_bucket": "medical-device-data",
    "input_key": "mdr/incidents_2025_q1.csv",
    "output_bucket": "medical-device-analysis",
    "output_prefix": "classifications/product/2025-04-09",
    "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "problem_type": "product",
    "batch_size": 2,
    "max_workers": 5,
    "rate_limit": 5,
    "imdrf_bucket": "medical-device-reference",
    "imdrf_key": "reference/imdrf.csv"
}

# 2. Sample JSON configuration for patient issue classification
PATIENT_JOB_CONFIG = {
    "input_bucket": "medical-device-data",
    "input_key": "mdr/incidents_2025_q1.csv", 
    "output_bucket": "medical-device-analysis",
    "output_prefix": "classifications/patient/2025-04-09",
    "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "problem_type": "patient",  # Patient problem classification
    "batch_size": 2,
    "max_workers": 5,
    "rate_limit": 5,
    "imdrf_bucket": "medical-device-reference",
    "imdrf_key": "reference/imdrf.csv"
}

# 3. AWS CLI commands to invoke the Lambda function

# For product issue classification
"""
aws lambda invoke \
    --function-name bedrock-batch-classifier \
    --cli-binary-format raw-in-base64-out \
    --payload '{"input_bucket":"medical-device-data","input_key":"mdr/incidents_2025_q1.csv","output_bucket":"medical-device-analysis","output_prefix":"classifications/product/2025-04-09","model_id":"anthropic.claude-3-5-sonnet-20240620-v1:0","problem_type":"product","batch_size":2,"max_workers":5,"rate_limit":5}' \
    product_response.json
"""

# For patient issue classification
"""
aws lambda invoke \
    --function-name bedrock-batch-classifier \
    --cli-binary-format raw-in-base64-out \
    --payload '{"input_bucket":"medical-device-data","input_key":"mdr/incidents_2025_q1.csv","output_bucket":"medical-device-analysis","output_prefix":"classifications/patient/2025-04-09","model_id":"anthropic.claude-3-5-sonnet-20240620-v1:0","problem_type":"patient","batch_size":2,"max_workers":5,"rate_limit":5}' \
    patient_response.json
"""

# 4. Example input CSV format
"""
id,mdr_text,date_received,product_code
1001,"Patient underwent surgery for placement of cardiac device. During procedure, the device was difficult to position and the physician had to make multiple attempts before successfully placing the device.",2025-01-15,ABC123
1002,"Patient reported skin irritation at the infusion site. Redness and swelling were observed around the area where the device was attached. The device was removed and symptoms resolved within 24 hours.",2025-01-16,XYZ456
1003,"During routine maintenance check, the medical staff found that the device was not functioning as expected. The display was showing error code E-42, which according to the user manual indicates a power supply issue.",2025-01-17,DEF789
"""

# 5. Example output formats

# Product classification output CSV format
"""
uid,mdr_text,input_tokens,output_tokens,ai_sufficient_info_annex_a_product,ai_assigned_annex_a_product,ai_supporting_labels_annex_a_product,ai_justification_annex_a_product,ai_justification_from_text_annex_a_product
1001,"Patient underwent surgery for placement of cardiac device. During procedure, the device was difficult to position and the physician had to make multiple attempts before successfully placing the device.",520,340,"true","Material Integrity Issue","[\"Component or Material Integrity Issue\",\"Implant Issue\"]","The device was difficult to position requiring multiple attempts, suggesting a potential issue with the design or material integrity of the device.","the device was difficult to position and the physician had to make multiple attempts before successfully placing the device"
"""

# Patient classification output CSV format
"""
uid,mdr_text,input_tokens,output_tokens,ai_sufficient_info_annex_e_patient,ai_assigned_annex_e_patient,ai_supporting_labels_annex_e_patient,ai_justification_annex_e_patient,ai_justification_from_text_annex_e_patient
1002,"Patient reported skin irritation at the infusion site. Redness and swelling were observed around the area where the device was attached. The device was removed and symptoms resolved within 24 hours.",485,362,"true","Skin Irritation","[\"Local Tissue Effects\",\"Inflammation\"]","The text describes clear signs of skin irritation (redness and swelling) at the infusion site where the device was attached, with symptoms resolving after device removal.","Redness and swelling were observed around the area where the device was attached"
"""

# 6. Example AWS Step Function state machine to process both product and patient classifications
STEP_FUNCTION_DEFINITION = {
    "Comment": "Medical Device Classification Pipeline",
    "StartAt": "ProcessProductIssues",
    "States": {
        "ProcessProductIssues": {
            "Type": "Task",
            "Resource": "arn:aws:states:::lambda:invoke",
            "Parameters": {
                "FunctionName": "arn:aws:lambda:us-east-1:123456789012:function:bedrock-batch-classifier",
                "Payload": {
                    "input_bucket": "medical-device-data",
                    "input_key": "mdr/incidents_2025_q1.csv",
                    "output_bucket": "medical-device-analysis",
                    "output_prefix": "classifications/product/2025-04-09",
                    "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "problem_type": "product",
                    "batch_size": 2,
                    "max_workers": 5,
                    "rate_limit": 5,
                    "imdrf_bucket": "medical-device-reference",
                    "imdrf_key": "reference/imdrf.csv"
                }
            },
            "Next": "ProcessPatientIssues"
        },
        "ProcessPatientIssues": {
            "Type": "Task",
            "Resource": "arn:aws:states:::lambda:invoke",
            "Parameters": {
                "FunctionName": "arn:aws:lambda:us-east-1:123456789012:function:bedrock-batch-classifier",
                "Payload": {
                    "input_bucket": "medical-device-data",
                    "input_key": "mdr/incidents_2025_q1.csv",
                    "output_bucket": "medical-device-analysis",
                    "output_prefix": "classifications/patient/2025-04-09",
                    "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "problem_type": "patient",
                    "batch_size": 2,
                    "max_workers": 5, 
                    "rate_limit": 5,
                    "imdrf_bucket": "medical-device-reference",
                    "imdrf_key": "reference/imdrf.csv"
                }
            },
            "Next": "MergeResults"
        },
        "MergeResults": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:123456789012:function:merge-classification-results",
            "Parameters": {
                "productResultsPrefix": "classifications/product/2025-04-09",
                "patientResultsPrefix": "classifications/patient/2025-04-09",
                "outputPrefix": "classifications/merged/2025-04-09",
                "s3Bucket": "medical-device-analysis"
            },
            "Next": "NotifyCompletion"
        },
        "NotifyCompletion": {
            "Type": "Task",
            "Resource": "arn:aws:states:::sns:publish",
            "Parameters": {
                "TopicArn": "arn:aws:sns:us-east-1:123456789012:classification-complete",
                "Message": "Medical device classification batch job complete. Product and patient results available in S3."
            },
            "End": true
        }
    }
}

# 7. Example CloudWatch scheduled event to run batch jobs weekly
CLOUDWATCH_EVENT_RULE = {
    "name": "weekly-mdr-classification",
    "description": "Triggers medical device classification every Monday",
    "schedule_expression": "cron(0 0 ? * MON *)",
    "state": "ENABLED",
    "targets": [
        {
            "id": "1",
            "arn": "arn:aws:states:us-east-1:123456789012:stateMachine:MedicalDeviceClassificationPipeline"
        }
    ]
}

# 8. Example batch job monitor in Python
MONITOR_SCRIPT = '''
import boto3
import json
import time
from datetime import datetime

def check_job_status(job_id, output_bucket, output_prefix):
    """Check the status of a batch job by looking for summary file"""
    s3 = boto3.client('s3')
    summary_key = f"{output_prefix}/job_summary.json"
    
    try:
        response = s3.get_object(Bucket=output_bucket, Key=summary_key)
        summary = json.loads(response['Body'].read().decode('utf-8'))
        return summary
    except s3.exceptions.NoSuchKey:
        return {"status": "IN_PROGRESS", "message": "Summary not found yet"}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def main():
    # Configuration 
    job_configs = [
        {
            "job_id": "product-classification-20250409",
            "output_bucket": "medical-device-analysis",
            "output_prefix": "classifications/product/2025-04-09"
        },
        {
            "job_id": "patient-classification-20250409",
            "output_bucket": "medical-device-analysis", 
            "output_prefix": "classifications/patient/2025-04-09"
        }
    ]
    
    # Monitor each job
    max_checks = 60  # Check for up to 60 minutes (1 minute intervals)
    checks = 0
    all_complete = False
    
    print(f"Starting job monitoring at {datetime.now().isoformat()}")
    
    while not all_complete and checks < max_checks:
        all_complete = True
        
        for config in job_configs:
            status = check_job_status(
                config["job_id"], 
                config["output_bucket"], 
                config["output_prefix"]
            )
            
            print(f"Job {config['job_id']}: {status.get('status', 'UNKNOWN')}")
            
            if status.get("status") != "COMPLETED" and status.get("status") != "COMPLETED_WITH_ERRORS":
                all_complete = False
        
        if not all_complete:
            print(f"Waiting for jobs to complete... (check {checks+1}/{max_checks})")
            time.sleep(60)  # Wait 1 minute before checking again
            checks += 1
    
    if all_complete:
        print("All jobs completed!")
    else:
        print("Monitoring timed out - some jobs may still be running")
    
    # Print final results
    for config in job_configs:
        final_status = check_job_status(
            config["job_id"], 
            config["output_bucket"], 
            config["output_prefix"]
        )
        print(f"Final status for {config['job_id']}:")
        print(json.dumps(final_status, indent=2))

if __name__ == "__main__":
    main()
'''