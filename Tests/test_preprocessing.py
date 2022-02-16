import pytest

from ..preprocess import *


data = read_data('chess.data')

def test_1d_coords():
    
    white_king_1d = normalise_x_y_coordinates(data, "king_x", "king_y")

    white_rook_1d = normalise_x_y_coordinates(data, 'rook_x', 'rook_y')

    black_king_1d = normalise_x_y_coordinates(data, 'king2_x', 'king2_y')

    data['white_king_1d'] = white_king_1d
    data['white_rook_1d'] = white_rook_1d
    data['black_king_1d'] = black_king_1d

    for i in range(len(data)):
        assert data['white_king_1d'][i] > -1
        assert data['white_rook_1d'][i] > -1
        assert data['black_king_1d'][i] > -1


def test_remove_missing_values():
    pass

