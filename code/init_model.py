import os
import shutil
from alpha_net import ChessNet
import torch

# Supprimer les logs existants
def remove_logs():
    dirs = ['./datasets/iter0/', './datasets/iter1/', './datasets/iter2/', './model_data/']
    for dir_path in dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

# Créer les nouveaux dossiers
def create_directories():
    dirs = ['./datasets/iter0/', './datasets/iter1/', './datasets/iter2/', './model_data/']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

# Initialiser et sauvegarder le réseau de neurones
def initialize_network():
    net = ChessNet()
    torch.save({'state_dict': net.state_dict()}, "./model_data/current_net_trained8_iter1.pth.tar")

if __name__ == "__main__":
    remove_logs()
    create_directories()
    initialize_network()
    print("Logs supprimés, dossiers créés et réseau initialisé avec succès.")
