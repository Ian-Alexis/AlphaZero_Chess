import os

# Définir la variable d'environnement XDG_RUNTIME_DIR
os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime'

# Désactiver le son dans pygame
os.environ['SDL_AUDIODRIVER'] = 'dsp'

import chess
import pygame
import threading

# Initialiser le plateau de jeu
board = chess.Board()

# Fonction pour dessiner le plateau avec les couleurs d'origine
def draw_board(screen):
    colors = [pygame.Color(238, 238, 210), pygame.Color(118, 150, 86)]
    for i in range(8):
        for j in range(8):
            color = colors[(i + j +1) % 2]
            pygame.draw.rect(screen, color, pygame.Rect((i) * 80, (7-j) * 80, 80, 80))

# Charger les images des pièces
def load_images():
    pieces = ['black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
              'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook']
    images = {}
    for piece in pieces:
        images[piece] = pygame.image.load(os.path.join('images', f'{piece}.png'))
    return images

# Dessiner les pièces sur le plateau
def draw_pieces(screen, board, images):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_image = images[piece_color_and_type(piece)]
            row, col = divmod(square, 8)
            screen.blit(piece_image, pygame.Rect((col) * 80, (7-row) * 80, 80, 80))

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
    # Inverser les couleurs des pièces
    color = 'white' if piece.color == chess.WHITE else 'black'
    return f'{color}_{piece_map[piece.piece_type]}'

# Fonction pour obtenir des mouvements de l'utilisateur
def get_move_input(board):
    while True:
        move = input("Entrez votre mouvement (ex. e4, Nf3): ").strip()
        try:
            board.push_san(move)
        except ValueError:
            print("Mouvement illégal ou invalide. Utilisez la notation FIDE (ex. e4, Nf3).")

# Fonction principale pour l'interface graphique
def main():
    pygame.init()
    screen = pygame.display.set_mode((640, 640))
    pygame.display.set_caption('Chess')

    images = load_images()
    clock = pygame.time.Clock()
    running = True

    # Démarrer un thread pour obtenir les mouvements de l'utilisateur
    input_thread = threading.Thread(target=get_move_input, args=(board,))
    input_thread.daemon = True
    input_thread.start()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw_board(screen)
        draw_pieces(screen, board, images)
        pygame.display.flip()
        clock.tick(30)  # Limiter les FPS à 30

    pygame.quit()


if __name__ == "__main__":
    main()
