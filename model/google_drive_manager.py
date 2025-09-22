import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

class GoogleDriveManager:
    def __init__(self, credentials_path='credentials.json'):
        self.credentials_path = credentials_path
        self.service = self.authenticate()

    def authenticate(self):
        SCOPES = ['https://www.googleapis.com/auth/drive']
        credentials = service_account.Credentials.from_service_account_file(
            self.credentials_path, scopes=SCOPES)
        service = build('drive', 'v3', credentials=credentials)
        return service

    def get_folder_id(self, folder_name, parent_id=None):
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        if not items:
            return None
        return items[0]['id']

    def list_files_in_folder(self, folder_id, extensions=None):
        query = f"'{folder_id}' in parents and trashed=false"
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        items = results.get('files', [])
        if extensions:
            items = [item for item in items if any(item['name'].lower().endswith(ext) for ext in extensions)]
        return items

    def download_file(self, file_id, destination_path=None):
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        if destination_path:
            with open(destination_path, 'wb') as f:
                f.write(fh.read())
            return destination_path
        return fh
