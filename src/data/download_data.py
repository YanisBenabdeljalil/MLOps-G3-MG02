import boto3
import os

BUCKET_NAME = "s3-g3mg02" 
LOCAL_FILE = os.path.join(os.path.dirname(__file__), "competency_job.csv")

def download_to_s3():
    """Charge le fichier situé dans src/data/ vers la zone brute de S3."""
    s3 = boto3.client('s3')
    
    if not os.path.exists(LOCAL_FILE):
        print(f"Erreur : Le fichier est introuvable à l'emplacement : {LOCAL_FILE}")
        return

    try:
        print(f"Envoi de {LOCAL_FILE} vers s3://{BUCKET_NAME}/raw/competency_job.csv...")
        s3.upload_file(LOCAL_FILE, BUCKET_NAME, "raw/competency_job.csv")
        print("Upload réussi dans la zone 'raw'.")
    except Exception as e:
        print(f"Erreur lors de l'upload : {e}")

if __name__ == "__main__":
    download_to_s3()