from model.google_drive_manager import GoogleDriveManager
from model.model_manager import ModelManager
# Importe aqui seu modelo UNet ou outros modelos

class MainController:
    def __init__(self):
        self.drive = GoogleDriveManager()
        # self.model_manager = ModelManager(SeuModelo())

    def baixar_dados(self, folder_id, destino):
        arquivos = self.drive.list_files_in_folder(folder_id)
        for arq in arquivos:
            self.drive.download_file(arq['id'], destination_path=f"{destino}/{arq['name']}")

    # Adicione métodos para orquestrar o pipeline, treinar, validar, etc.
