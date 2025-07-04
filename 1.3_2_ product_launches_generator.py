###### This is a Python script used on Google Cloud Platform for generating product launches
# using newest Gemini model at the time.

import google.generativeai as genai
import vertexai
from vertexai.generative_models import GenerativeModel
import pandas as pd
import json
import time
import os
import re
import asyncio
import concurrent.futures
from google.cloud import storage
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

# Setting up Vertex AI
PROJECT_ID = "#################################"
BUCKET_NAME = "censor_bucket"
REGION = "us-central1"

vertexai.init(project=PROJECT_ID, location=REGION)
genai_client = GenerativeModel("gemini-2.5-pro-preview-03-25")

# GCS functions
def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    try:
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {source_blob_name} to {destination_file_name}")
        return True
    except Exception as e:
        print(f"Error downloading {source_blob_name}: {e}")
        return False

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to {destination_blob_name}")

# Progress tracking
PROGRESS_GCS_PATH = "product_launches/progress.json"
BATCH_SIZE = 25  # tickers per batch

def load_progress():
    temp_progress_file = "progress.json"
    if download_from_gcs(BUCKET_NAME, PROGRESS_GCS_PATH, temp_progress_file):
        with open(temp_progress_file, 'r') as f:
            progress = json.load(f).get('last_index', -1)
        print(f"Loaded progress: last_index = {progress}")
        os.remove(temp_progress_file)
        return progress
    else:
        print("Progress file not found in GCS. Creating a new one.")
        default_progress = {'last_index': -1}
        with open(temp_progress_file, 'w') as f:
            json.dump(default_progress, f)
        upload_to_gcs(BUCKET_NAME, temp_progress_file, PROGRESS_GCS_PATH)
        os.remove(temp_progress_file)
        return -1

def save_progress(index):
    temp_progress_file = "progress.json"
    with open(temp_progress_file, 'w') as f:
        json.dump({'last_index': index}, f)
    upload_to_gcs(BUCKET_NAME, temp_progress_file, PROGRESS_GCS_PATH)
    os.remove(temp_progress_file)

# Validating event
def validate_event(event, ticker):
    try:
        if not all(key in event for key in ["Ticker", "Date", "Time", "Event_Type", "Description", "Source"]):
            return False
        if event["Ticker"] != ticker or event["Event_Type"] not in ["Product Launch", "Service Announcement", "Partnership/Milestone"]:
            return False
        date = datetime.strptime(event["Date"], "%Y-%m-%d")
        start_date = datetime(2020, 3, 6)
        end_date = datetime(2025, 2, 15)
        if not (start_date <= date <= end_date):
            return False
        time_obj = time.strptime(event["Time"], "%H:%M:%S")
        if not (8 <= time_obj.tm_hour <= 17):
            return False
        # Validating source (URL or reference format)
        if not (event["Source"].startswith("http") or re.match(r"[A-Za-z\s]+, \d{4}-\d{2}-\d{2}", event["Source"])):
            return False
        # Ensuring concise description
        if len(event["Description"].split()) > 50:
            return False
        return True
    except (ValueError, KeyError):
        return False

# Generating prompt
def generate_prompt(ticker):
    return (
        f"For the company with ticker {ticker}, generate a list of all known corporate events "
        f"(product launches, service announcements, partnerships/milestones) from March 6, 2020, to February 15, 2025. "
        f"Follow the system instructions:\n"
        "You are a financial analyst specializing in corporate events for NASDAQ companies. Your task is to generate a list of verified corporate events "
        "(product launches, service announcements, partnerships/milestones) for a given ticker from March 6, 2020, to February 15, 2025, "
        "that are likely to cause significant stock price volatility (>5% daily price change). Follow these rules:\n"
        "1. **Event Types**:\n"
        "   - **Product Launch**: Public introduction of a new tangible product (e.g., hardware, software, drugs). Example: Apple’s iPhone 15 (AAPL, 2023-09-12).\n"
        "   - **Service Announcement**: Launch of a new service or significant update (e.g., new airline routes, app features). Example: American Airlines’ new routes (AAL, 2021-03-29).\n"
        "   - **Partnership/Milestone**: Major partnerships, clinical trial results, or regulatory approvals. Example: AbCellera’s NIH collaboration (ABCL, 2020-06-29).\n"
        "2. Return results as a JSON list of dictionaries with fields: \"Ticker\" (string), \"Date\" (YYYY-MM-DD), \"Time\" (HH:MM:SS, estimate if unknown, e.g., 08:00:00 for pre-market), "
        "\"Event_Type\" (one of \"Product Launch\", \"Service Announcement\", \"Partnership/Milestone\"), \"Description\" (brief, max 50 words), "
        "\"Source\" (URL or reference, e.g., “Apple Newsroom, 2023-09-12”).\n"
        "3. Include only events verified by credible sources (e.g., SEC filings, company press releases, reputable news like Reuters, Bloomberg). "
        "Exclude speculative, unconfirmed, or minor events (e.g., routine updates, events beyond April 22, 2025).\n"
        "4. If no verified events are found, return an empty list: [].\n"
        "5. Ensure dates are within March 6, 2020, to February 15, 2025, and times are within business hours (08:00:00–17:00:00).\n"
        "6. Prioritize events likely to cause >5% stock price volatility (e.g., major product releases, FDA approvals).\n"
        "7. Do not fabricate or hallucinate events; only include verified events. It is acceptable to return no events if none are found.\n"
        "Example Output:\n"
        "[\n"
        "    {\"Ticker\": \"AAPL\", \"Date\": \"2023-09-12\", \"Time\": \"13:00:00\", \"Event_Type\": \"Product Launch\", \"Description\": \"iPhone 15 and Apple Watch Series 9 announced\", \"Source\": \"Apple Newsroom, 2023-09-12\"},\n"
        "    {\"Ticker\": \"AAL\", \"Date\": \"2021-03-29\", \"Time\": \"09:00:00\", \"Event_Type\": \"Service Announcement\", \"Description\": \"New route expansions post-COVID\", \"Source\": \"PR Newswire, 2021-03-29\"},\n"
        "    {\"Ticker\": \"ABCL\", \"Date\": \"2020-06-29\", \"Time\": \"09:00:00\", \"Event_Type\": \"Partnership/Milestone\", \"Description\": \"Collaboration with NIH for SARS-CoV-2 antibodies\", \"Source\": \"Business Wire, 2020-06-29\"}\n"
        "]"
    )

# Synchronous function for generating events for a single ticker
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=60))
def generate_events(ticker):
    try:
        prompt = generate_prompt(ticker)
        response = genai_client.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}]
        )
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        result = json.loads(text)
        if not isinstance(result, list):
            print(f"Invalid response format for {ticker}: {text}")
            return []
        valid_events = [e for e in result if validate_event(e, ticker)]
        return valid_events
    except Exception as e:
        if "quota" in str(e).lower() or "rate limit" in str(e).lower():
            print(f"Rate limit hit for {ticker}. Retrying...")
            raise  # Trigger retry
        print(f"Error for {ticker}: {e}")
        return []

# Processing a batch of tickers synchronously
def process_batch(tickers):
    product_launches = []
    for ticker in tickers:
        events = generate_events(ticker)
        product_launches.extend(events)
        print(f"Processed {ticker}: {len(events)} events found")
    return product_launches

def process_and_upload_batches(df, batch_size):
    total_tickers = len(df)
    start_index = load_progress() + 1

    if start_index >= total_tickers:
        print("All tickers already processed!")
        return

    for batch_start in range(start_index, total_tickers, batch_size):
        batch_end = min(batch_start + batch_size, total_tickers)
        batch_tickers = df.iloc[batch_start:batch_end]['Ticker'].tolist()
        print(f"Processing batch: tickers {batch_start + 1} to {batch_end}")

        product_launches = process_batch(batch_tickers)

        # Saving batch output
        temp_file = f"temp_batch_{batch_start}_{batch_end - 1}.json"
        with open(temp_file, 'w') as f:
            json.dump(product_launches, f, indent=4)
        gcs_path = f"product_launches/batch_{batch_start}_{batch_end - 1}.json"
        upload_to_gcs(BUCKET_NAME, temp_file, gcs_path)
        os.remove(temp_file)

        save_progress(batch_end - 1)
        print(f"Batch processed and uploaded: {batch_end}/{total_tickers} ({batch_end / total_tickers * 100:.2f}%)")

# Merging batches
def merge_batches_to_dataframe():
    batch_files = download_files(BUCKET_NAME, "product_launches/batch_", "batch_")
    if not batch_files:
        print("No batch files found to merge!")
        return None
    merged_output_file = "merged_product_launches.json"
    merged_df = merge_json_files(batch_files, merged_output_file)
    upload_to_gcs(BUCKET_NAME, merged_output_file, "product_launches/merged_product_launches.json")
    cleanup_temp_files(batch_files, "temp_batches", merged_output_file)
    return merged_df

def download_files(bucket_name, prefix, file_pattern):
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    if not os.path.exists("temp_batches"):
        os.makedirs("temp_batches")
    blobs = bucket.list_blobs(prefix=prefix)
    downloaded_files = []
    for blob in blobs:
        if not blob.name.endswith('.json') or file_pattern not in blob.name:
            continue
        local_file = os.path.join("temp_batches", os.path.basename(blob.name))
        blob.download_to_filename(local_file)
        downloaded_files.append(local_file)
        print(f"Downloaded {blob.name} to {local_file}")
    def extract_start_index(filename):
        match = re.search(r'batch_(\d+)_\d+\.json', os.path.basename(filename))
        return int(match.group(1)) if match else float('inf')
    downloaded_files.sort(key=extract_start_index)
    return downloaded_files

def merge_json_files(file_list, output_file):
    merged_data = []
    for file in file_list:
        with open(file, 'r') as f:
            batch_data = json.load(f)
            if isinstance(batch_data, list):
                merged_data.extend(batch_data)
            else:
                print(f"Warning: {file} does not contain a list, skipping.")
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)
    print(f"Merged {len(file_list)} files into {output_file} with {len(merged_data)} records.")
    return pd.DataFrame(merged_data)

def cleanup_temp_files(file_list, temp_dir, merged_file):
    for file in file_list:
        os.remove(file)
    os.rmdir(temp_dir)
    if os.path.exists(merged_file):
        os.remove(merged_file)
    print(f"Cleaned up temporary files and directory: {temp_dir}")


# Loading ticker dataset
TICKER_GCS_PATH = "product_launches/ticker_dataset.json"
local_ticker_file = "ticker_dataset.json"
if not download_from_gcs(BUCKET_NAME, TICKER_GCS_PATH, local_ticker_file):
    raise FileNotFoundError("Ticker dataset not found in GCS")

with open(local_ticker_file, 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)
print(f"Loaded {len(df)} tickers")

# Running and merging
process_and_upload_batches(df, BATCH_SIZE)
merged_df = merge_batches_to_dataframe()
