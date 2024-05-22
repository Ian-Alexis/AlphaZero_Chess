# git clone https://github.com/geochri/AlphaZero_Chess.git
# cd 
# pip install -r requirements.txt

import torch
from alpha_net import ChessNet  # Assurez-vous que ChessNet est correctement défini dans alpha_net

# Initialisation du réseau
alpha_net = ChessNet()

# Sauvegarder le réseau initialisé aléatoirement avec le nom attendu par pipeline.py
torch.save({'state_dict': alpha_net.state_dict()}, './model_data/current_net_trained8_iter1.pth.tar')


# python alpha_net.py

# mkdir -p ./model_data
# mkdir -p ./datasets/iter0
# mkdir -p ./datasets/iter1
# mkdir -p ./datasets/iter2


# python pipeline.py

