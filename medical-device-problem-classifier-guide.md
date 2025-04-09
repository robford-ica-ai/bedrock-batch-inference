# AWS Bedrock Batch Inference Implementation Guide

This guide outlines how to implement and deploy the batch inference script for processing large volumes of medical device incident texts through Claude models via AWS Bedrock.

## Overview

The solution performs automated classification of medical device incidents into IMDRF (International Medical Device Regulators Forum) categories using AWS Bedrock and Claude models. It supports:

- Classification of both product and patient issues
- Batch processing of large datasets
- Parallel processing with configurable concurrency
- Handling rate limits and throttling with exponential backoff
- Cost estimation and tracking
- S3 integration for input/output

## Components

- **Lambda Function:** Core processing engine that handles batch jobs
- **S3 Buckets:** Store input data, reference data (IMDRF), and output results
- **AWS Bedrock:** Provides access to Claude models for classification

## Deployment Steps

### 1. Upload Reference Data

Upload your IMDRF reference data to an S3 bucket:

```shell
aws s3 cp imdrf.csv s3://your-bucket/reference/imdrf.csv
```

### 2. Create Lambda Function

1. Create a new Lambda function using Python 3.10+ runtime
2. Set appropriate memory (min 1024MB) and timeout (10+ minutes)
3. Add necessary permissions:
   - S3 read/write access
   - Bedrock invoke model permission

### 3. Deploy Code

Upload the batch_inference_script.py to your Lambda function:

```shell
zip -r lambda_package.zip batch_inference_script.py
aws lambda update-function-code --function-name bedrock-batch-classifier --zip-file fileb://lambda_package.zip
```

### 4. Configure Environment Variables

Set these environment variables in your Lambda function:

- `DEFAULT_MODEL_ID`: Default model to use (e.g., "anthropic.claude-3-5-sonnet-20240620-v1:0")
- `DEFAULT_BATCH_SIZE`: Number of texts to process in one batch (recommended: 1-3)
- `DEFAULT_MAX_WORKERS`: Maximum concurrent API calls (recommended: 5)
- `DEFAULT_RATE_LIMIT`: Maximum API calls per second (recommended: 5)

### 5. Add Required Packages

Create a Lambda layer with these dependencies:

- pandas
- numpy
- boto3 (Latest version)

## Usage

### Input Format

The input CSV file should have at minimum these columns:

- `id`: Unique identifier for each record
- `mdr_text`: The medical device incident text to classify

### Triggering Batch Jobs

Invoke the Lambda function with a payload like this:

```json
{
    "input_bucket": "your-data-bucket",
    "input_key": "inputs/mdr_records.csv",
    "output_bucket": "your-output-bucket",
    "output_prefix": "outputs/job-20240409",
    "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "problem_type": "product",
    "batch_size": 2,
    "max_workers": 5,
    "rate_limit": 5,
    "imdrf_bucket": "your-reference-bucket",
    "imdrf_key": "reference/imdrf.csv"
}
```

You can trigger this via:

```shell
aws lambda invoke --function-name bedrock-batch-classifier \
    --payload file://job-config.json \
    output.json
```

### Output Format

The solution produces:

1. CSV files for each processed chunk
2. A summary JSON file with job statistics

## Cost Management

The script includes cost estimation based on token usage. Current pricing (subject to change):

- Input tokens: $3.60 per million tokens
- Output tokens: $18.00 per million tokens

## Monitoring

Monitor batch jobs through:

1. CloudWatch Logs for Lambda execution details
2. Job summary files in the output S3 location

## Best Practices

1. **Batch Size:** Keep batch sizes small (1-3 texts per batch) for optimal performance
2. **Rate Limiting:** Respect Bedrock service limits (start with 5 RPS)
3. **Chunking:** Process large datasets in chunks to avoid memory issues
4. **Testing:** Start with small samples before processing full datasets

## Troubleshooting

Common issues and solutions:

- **Throttling errors:** Reduce concurrent workers or rate limit
- **Memory errors:** Reduce chunk size
- **Timeout errors:** Increase Lambda timeout setting
- **Parse errors:** Check for invalid text formats in your input data

## Additional Resources

For more information about AWS Bedrock, visit the [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/).
