from google.cloud import storage

def get_from_cloud_storage(cloud_bucket, filename):
    client = storage.Client()
    blob = client.bucket(cloud_bucket).blob(filename)
    return blob.download_as_bytes()