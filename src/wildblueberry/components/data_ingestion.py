import os
import urllib.request as request
import zipfile
from pathlib import Path
import subprocess
import shutil
from wildblueberry import logger
from wildblueberry.utils import get_size
from wildblueberry.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_file(self):
        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            # Ensure kaggle command exists
            kaggle_check = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
            if kaggle_check.returncode != 0:
                logger.error("Kaggle CLI not found. Ensure it's installed and accessible in your PATH.")
                return

            # Handle `datasets` downloads
            if dataset_url.split('/')[3] == 'datasets':
                if not os.path.isfile("artifacts/data_ingestion/data.zip"):
                    dataset_command = [
                        "kaggle", "datasets", "download", "-d",
                        f"{dataset_url.split('/')[4]}/{dataset_url.split('/')[5]}"
                    ]
                    result = subprocess.run(dataset_command, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.error(f"Dataset download failed: {result.stderr}")
                        return
                    
                    # Check and move downloaded file
                    if os.path.isfile("archive.zip"):
                        shutil.move("archive.zip", "artifacts/data_ingestion/data.zip")
                    elif os.path.isfile(f"{dataset_url.split('/')[5]}.zip"):
                        shutil.move(f"{dataset_url.split('/')[5]}.zip", "artifacts/data_ingestion/data.zip")
                    else:
                        logger.error("Download failed or file not found after download.")
            
            # Handle `competitions` downloads
            else:
                if not os.path.isfile("artifacts/data_ingestion/data.zip"):
                    competition_command = [
                        "kaggle", "competitions", "download", "-c",
                        dataset_url.split('/')[4]
                    ]
                    result = subprocess.run(competition_command, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.error(f"Competition download failed: {result.stderr}")
                        return
                    
                    # Move the competition file
                    file_name = f"{dataset_url.split('/')[4]}.zip"
                    if os.path.isfile(file_name):
                        try:
                            shutil.move(file_name, "artifacts/data_ingestion/data.zip")
                            logger.info("File moved successfully.")
                        except Exception as e:
                            logger.error(f"Error moving file: {e}")
                    else:
                        logger.error("Competition file download failed or file not found after download.")

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Subprocess error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during download: {e}")
            raise e

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
  
