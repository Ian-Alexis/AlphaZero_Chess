# git clone https://github.com/geochri/AlphaZero_Chess.git
# cd 
# pip install -r requirements.txt

# import torch
# from alpha_net import ChessNet  # Assurez-vous que ChessNet est correctement défini dans alpha_net

# # Initialisation du réseau
# alpha_net = ChessNet()

# # Sauvegarder le réseau initialisé aléatoirement avec le nom attendu par pipeline.py
# torch.save({'state_dict': alpha_net.state_dict()}, './model_data/current_net_trained8_iter1.pth.tar')


# python alpha_net.py

# mkdir -p ./model_data
# mkdir -p ./datasets/iter0
# mkdir -p ./datasets/iter1
# mkdir -p ./datasets/iter2


# En cas de problème avec CUDA
#pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/test/cu118


# python pipeline.py


import os

# Récupérer le nom d'utilisateur
username = os.getlogin()
print(username)
