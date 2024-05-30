import numpy as np
import chess

def encode_board(board):
    encoded = np.zeros([8, 8, 22]).astype(int)
    encoder_dict = {
        chess.PAWN: 5, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 0,
        chess.QUEEN: 3, chess.KING: 4
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color_offset = 0 if piece.color == chess.WHITE else 6
            encoded[square // 8, square % 8, encoder_dict[piece.piece_type] + color_offset] = 1
    
    if board.turn == chess.WHITE:
        encoded[:, :, 12] = 1  # White to move
    if board.has_kingside_castling_rights(chess.WHITE):
        encoded[:, :, 14] = 1  # White can castle kingside
    if board.has_queenside_castling_rights(chess.WHITE):
        encoded[:, :, 13] = 1  # White can castle queenside
    if board.has_kingside_castling_rights(chess.BLACK):
        encoded[:, :, 16] = 1  # Black can castle kingside
    if board.has_queenside_castling_rights(chess.BLACK):
        encoded[:, :, 15] = 1  # Black can castle queenside

    encoded[:, :, 17] = board.fullmove_number
    encoded[:, :, 18] = board.halfmove_clock
    
    return encoded

def decode_action(board, encoded):
    encoded_a = np.zeros([4672])
    encoded_a[encoded] = 1
    encoded_a = encoded_a.reshape(8, 8, 73)
    a, b, c = np.where(encoded_a == 1)
    i_pos, f_pos, prom = [], [], []
    for pos in zip(a, b, c):
        i, j, k = pos
        initial_pos = chess.square(j, i)
        promoted = None
        if 0 <= k <= 13:
            dy = 0
            if k < 7:
                dx = k - 7
            else:
                dx = k - 6
            final_pos = chess.square(j, i + dx)
        elif 14 <= k <= 27:
            dx = 0
            if k < 21:
                dy = k - 21
            else:
                dy = k - 20
            final_pos = chess.square(j + dy, i)
        elif 28 <= k <= 41:
            if k < 35:
                dy = k - 35
            else:
                dy = k - 34
            dx = dy
            final_pos = chess.square(j + dy, i + dx)
        elif 42 <= k <= 55:
            if k < 49:
                dx = k - 49
            else:
                dx = k - 48
            dy = -dx
            final_pos = chess.square(j + dy, i + dx)
        elif 56 <= k <= 63:
            knight_moves = [(2, -1), (2, 1), (1, -2), (-1, -2), (-2, 1), (-2, -1), (-1, 2), (1, 2)]
            dx, dy = knight_moves[k - 56]
            final_pos = chess.square(j + dy, i + dx)
        else:
            promotion_pieces = ["r", "n", "b"]
            if k in range(64, 73):
                if k < 67:
                    dx = 1 if board.turn == chess.WHITE else -1
                else:
                    dx = 1 if board.turn == chess.WHITE else -1
                    dy = 1 if (k - 67) % 2 == 0 else -1
                if k in [64, 65, 66]:
                    dy = 0
                promoted = promotion_pieces[k % 3]
                final_pos = chess.square(j + dy, i + dx)
        if board.piece_at(initial_pos) == chess.PAWN and chess.square_rank(final_pos) in [0, 7] and promoted is None:
            promoted = "q" if board.turn == chess.WHITE else "Q"
        i_pos.append(initial_pos)
        f_pos.append(final_pos)
        prom.append(promoted)
    return i_pos, f_pos, prom

def encode_action(board, initial_pos, final_pos, underpromote=None):
    encoded = np.zeros([8, 8, 73]).astype(int)
    i, j = chess.square_rank(initial_pos), chess.square_file(initial_pos)
    x, y = chess.square_rank(final_pos), chess.square_file(final_pos)
    dx, dy = x - i, y - j
    piece = board.piece_at(initial_pos)
    if piece and piece.piece_type in [chess.ROOK, chess.BISHOP, chess.QUEEN, chess.KING, chess.PAWN]:
        if dx != 0 and dy == 0:
            if dx < 0:
                idx = 7 + dx
            else:
                idx = 6 + dx
        elif dx == 0 and dy != 0:
            if dy < 0:
                idx = 21 + dy
            else:
                idx = 20 + dy
        elif dx == dy:
            if dx < 0:
                idx = 35 + dx
            else:
                idx = 34 + dx
        elif dx == -dy:
            if dx < 0:
                idx = 49 + dx
            else:
                idx = 48 + dx
    elif piece and piece.piece_type == chess.KNIGHT:
        knight_moves = [(2, -1), (2, 1), (1, -2), (-1, -2), (-2, 1), (-2, -1), (-1, 2), (1, 2)]
        idx = 56 + knight_moves.index((dx, dy))
    elif piece and piece.piece_type == chess.PAWN and (x == 0 or x == 7) and underpromote is not None:
        promotion_pieces = ["r", "n", "b"]
        if abs(dx) == 1 and dy == 0:
            idx = 64 + promotion_pieces.index(underpromote)
        elif abs(dx) == 1 and dy == -1:
            idx = 67 + promotion_pieces.index(underpromote)
        elif abs(dx) == 1 and dy == 1:
            idx = 70 + promotion_pieces.index(underpromote)
    encoded[i, j, idx] = 1
    encoded = encoded.reshape(-1)
    encoded = np.where(encoded == 1)[0][0]
    return encoded
