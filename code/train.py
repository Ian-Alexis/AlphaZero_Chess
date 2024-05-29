import os
import pickle
import numpy as np
import torch
from alpha_net import ChessNet, train

def train_chessnet(net_to_train="current_net_trained8_iter1.pth.tar", save_as="current_net_trained9_iter1.pth.tar"):
    # gather data
    board_states = []
    policies = []
    values = []
    
    data_path = "./datasets/iter2/"
    for idx, file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path, file)
        with open(filename, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
            for state, policy, value in data:
                board_states.append(state)
                policies.append(policy)
                values.append(value)
    
    # Si nécessaire, ajouter un autre chemin pour charger d'autres datasets
    data_path = "./datasets/iter1/"
    for idx, file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path, file)
        with open(filename, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
            for state, policy, value in data:
                board_states.append(state)
                policies.append(policy)
                values.append(value)
    
    # Convertir les listes en tableaux NumPy
    board_states = np.array(board_states)
    policies = np.array(policies)
    values = np.array(values)
    
    print("Loaded data example:", board_states[0], policies[0], values[0])
    print("Shapes:", board_states.shape, policies.shape, values.shape)
    
    # Créer le dataset final
    datasets = list(zip(board_states, policies, values))
    
    # train net
    net = ChessNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    current_net_filename = os.path.join("./model_data/", net_to_train)
    checkpoint = torch.load(current_net_filename)
    net.load_state_dict(checkpoint['state_dict'])
    train(net, datasets)
    # save results
    torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/", save_as))

if __name__ == "__main__":
    train_chessnet()
