import boto3
import io
import pandas as pd

BUCKET_NAME = "s3-g3mg02"

def save_processed_data(df: pd.DataFrame, filename: str):
    """Sauvegarde le DataFrame nettoyé dans la zone 'processed' de S3."""
    s3 = boto3.client('s3')
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    try:
        s3.put_object(
            Bucket=BUCKET_NAME, 
            Key=f"processed/{filename}", 
            Body=csv_buffer.getvalue()
        )
        print(f"Sauvegarde réussie : s3://{BUCKET_NAME}/processed/{filename}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde : {e}")