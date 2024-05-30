import os
import chess
import pygame
import threading
import torch
from alpha_net import ChessNet
from MCTS_chess2 import UCT_search
import encoder_decoder2 as ed2
import ctypes

# Définir la variable d'environnement XDG_RUNTIME_DIR
os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime'

# Désactiver le son dans pygame
os.environ['SDL_AUDIODRIVER'] = 'dsp'

# Charger le modèle entraîné
model_path = "./model_data/current_net_trained9_iter1.pth.tar"
model = ChessNet()
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

if torch.cuda.is_available():
    model.cuda()

# Initialiser le plateau de jeu
board = chess.Board()

# Fonction pour dessiner le plateau avec les couleurs d'origine
def draw_board(screen):
    colors = [pygame.Color(238, 238, 210), pygame.Color(118, 150, 86)]
    for i in range(8):
        for j in range(8):
            color = colors[(i + j + 1) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(i * 80, (7 - j) * 80, 80, 80))

# Charger les images des pièces
def load_images():
    pieces = ['black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
              'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook']
    images = {}
    for piece in pieces:
        try:
            images[piece] = pygame.image.load(os.path.join('images', f'{piece}.png'))
        except pygame.error as e:
            print(f"Erreur lors du chargement de l'image {piece}: {e}")
    return images

# Dessiner les pièces sur le plateau
def draw_pieces(screen, board, images):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_image = images[piece_color_and_type(piece)]
            row, col = divmod(square, 8)
            screen.blit(piece_image, pygame.Rect(col * 80, (7 - row) * 80, 80, 80))

# Modifier la fonction pour inverser les couleurs des pièces
def piece_color_and_type(piece):
    piece_map = {
        chess.PAWN: 'pawn',
        chess.KNIGHT: 'knight',
        chess.BISHOP: 'bishop',
        chess.ROOK: 'rook',
        chess.QUEEN: 'queen',
        chess.KING: 'king'
    }
    color = 'white' if piece.color == chess.WHITE else 'black'
    return f'{color}_{piece_map[piece.piece_type]}'

# Fonction pour obtenir des mouvements de l'utilisateur
def get_move_input(board):
    while True:
        move = input("Entrez votre mouvement (ex. e4, Nf3): ").strip()
        try:
            board.push_san(move)
            break
        except ValueError:
            print("Mouvement illégal ou invalide. Utilisez la notation FIDE (ex. e4, Nf3).")

# Fonction pour obtenir des mouvements du modèle
def get_model_move(board):
    game_state = ed2.encode_board(board)
    game_state = torch.from_numpy(game_state).float().cuda().unsqueeze(0)
    best_move_idx, _ = UCT_search(board, 111, model)
    initial_pos, final_pos, promotion = ed2.decode_action(board, best_move_idx)

    # Convertir les positions en notation UCI
    initial_uci = chess.square_name(initial_pos[0])
    final_uci = chess.square_name(final_pos[0])
    promotion_piece = promotion[0] if promotion[0] else ''

    move = chess.Move.from_uci(f"{initial_uci}{final_uci}{promotion_piece}")
    return move

# Fonction pour vérifier et afficher le résultat de la partie
def check_game_status(board):
    if board.is_checkmate():
        winner = "Noir" if board.turn == chess.WHITE else "Blanc"
        print(f"Checkmate! Les {winner}s gagnent.")
        return True
    elif board.is_stalemate():
        print("Pat! La partie est nulle.")
        return True
    elif board.is_insufficient_material():
        print("Matériel insuffisant pour mater. La partie est nulle.")
        return True
    elif board.can_claim_fifty_moves():
        print("Règle des 50 coups atteinte. La partie est nulle.")
        return True
    elif board.can_claim_threefold_repetition():
        print("Répétition triple atteinte. La partie est nulle.")
        return True
    return False

# Fonction pour amener la fenêtre au premier plan
def bring_window_to_front():
    hwnd = pygame.display.get_wm_info()['window']
    ctypes.windll.user32.SetForegroundWindow(hwnd)

# Fonction principale pour l'interface graphique
def main():
    pygame.init()
    screen = pygame.display.set_mode((640, 640))
    pygame.display.set_caption('Chess')

    images = load_images()
    clock = pygame.time.Clock()
    running = True

    while running:
        bring_window_to_front()  # Amener la fenêtre au premier plan

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_board(screen)
        draw_pieces(screen, board, images)
        pygame.display.flip()

        if check_game_status(board):
            running = False
            continue

        if board.turn == chess.WHITE:
            get_move_input(board)
        else:
            model_move = get_model_move(board)
            board.push(model_move)

        clock.tick(30)  # Limiter les FPS à 30

    pygame.quit()

if __name__ == "__main__":
    main()
