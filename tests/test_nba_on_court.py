import numpy as np
import pandas as pd
import pytest

from nba_on_court import __version__
from nba_on_court import nba_on_court
from requests import ConnectionError


def test_version():
    assert __version__ == '0.1.1'


def test_players_start_quater(ten_players_period):
    start_players = nba_on_court.players_on_court(ten_players_period)\
                        .iloc[0, 34:].reset_index(drop=True)
    assert all(start_players == pd.Series(
        [202954, 1627737, 1629673, 203952, 1629672,
         1627752, 1627747, 1629013, 201145, 1628386]))


def test_sort_players(ten_players_period):
    unsort_pl = nba_on_court.players_in_quater(ten_players_period)
    sort_pl = nba_on_court.sort_players(ten_players_period, unsort_pl)
    assert sort_pl == [202954, 1627737, 1629673, 203952, 1629672,
                       1627752, 1627747, 1629013, 201145, 1628386]


def test_nine_players(nine_players_period):
    player_id = nba_on_court.players_in_quater(nine_players_period)
    assert (202066 not in player_id) and (len(player_id) == 9)


def test_lost_internet(nine_players_period):
    with pytest.raises(ConnectionError):
        nba_on_court.players_on_court(nine_players_period,
                                               timeout=0.00001)


def test_eleven_players(eleven_players_period):
    player_id = nba_on_court.players_in_quater(eleven_players_period)
    assert (202326 in player_id) and (len(player_id) == 11)


def test_cashe():
    info1 = nba_on_court._cache_player_info()
    info2 = nba_on_court._cache_player_info()
    assert info1.__hash__() == info2.__hash__()


@pytest.mark.parametrize(
    ('all_id', 'shape'), [
        ([202954, 1627737, 1629673, 203952, 1629672, 1627752,
          1627747, 1629013, 201145, 1628386], (124, 44)),
        (np.array([202954, 1627737, 1629673, 203952, 1629672,
                   1627752, 1627747, 1629013, 201145, 1628386]), (124, 44))
    ]
)
def test_fill_list_array(ten_players_period, all_id, shape):
    df = nba_on_court.fill_columns(ten_players_period, all_id)
    assert df.shape == shape


def test_replace(ten_players_period):
    df = nba_on_court.players_on_court(ten_players_period)
    pl_list = nba_on_court\
        .players_name(df.iloc[0, 34:].reset_index(drop=True))
    assert pl_list == ['Brad Wanamaker', 'Marquese Chriss', 'Jordan Poole',
                       'Andrew Wiggins', 'Eric Paschall', 'Taurean Prince',
                       'Caris LeVert', 'Landry Shamet', 'Jeff Green',
                       'Jarrett Allen']
