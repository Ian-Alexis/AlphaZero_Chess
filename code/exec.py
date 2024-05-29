# # git clone https://github.com/geochri/AlphaZero_Chess.git
# # cd 
# # pip install -r requirements.txt

# import torch
# import os
# from alpha_net import ChessNet

# net = ChessNet()
# torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/", "current_net.pth.tar"))

# import os
# import pickle

# # Chemin du fichier pickle
# file_path = r'code\datasets\iter2\dataset_cpu0_0_2024-05-28'

# # Vérification de l'existence du fichier
# if os.path.isfile(file_path):
#     # Obtenir la taille du fichier
#     file_size = os.path.getsize(file_path)
#     # Obtenir l'extension du fichier
#     file_extension = os.path.splitext(file_path)[1]

#     print(f"Taille du fichier: {file_size} octets")
#     print(f"Extension du fichier: {file_extension}")

#     # Ouvrir et lire le fichier pickle
#     try:
#         with open(file_path, 'rb') as file:
#             data = pickle.load(file)
#         print("Le fichier pickle a été chargé avec succès.")
#         # Afficher un aperçu du contenu du fichier pickle
#         if isinstance(data, list):
#             print("Aperçu du contenu du fichier:", data[:5])
#         else:
#             print("Aperçu du contenu du fichier:", data)
#     except pickle.UnpicklingError:
#         print("Erreur lors du chargement du fichier pickle.")
# else:
#     print("Le fichier pickle n'a pas été trouvé.")




# # # Initialisation du réseau
# # alpha_net = ChessNet()

# # # Sauvegarder le réseau initialisé aléatoirement avec le nom attendu par pipeline.py
# # torch.save({'state_dict': alpha_net.state_dict()}, './model_data/current_net_trained8_iter1.pth.tar')


# # python alpha_net.py

# # mkdir -p ./model_data
# # mkdir -p ./datasets/iter0
# # mkdir -p ./datasets/iter1
# # mkdir -p ./datasets/iter2


# # En cas de problème avec CUDA
# #pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/test/cu118


# # python pipeline.py


