from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_on_court import nba_on_court, __version__, hexdata
from requests import ConnectionError


def test_version():
    assert __version__ == '0.3.0'


def test_players_start_quater(ten_players_period):
    start_players = nba_on_court.players_on_court(ten_players_period)\
                        .iloc[0, 34:].reset_index(drop=True)
    assert all(start_players == pd.Series(
        [202954, 1627737, 1629673, 203952, 1629672,
         1627752, 1627747, 1629013, 201145, 1628386]))


def test_sort_players(ten_players_period):
    unsort_pl = nba_on_court._players_in_quater(ten_players_period)
    sort_pl = nba_on_court._sort_players(ten_players_period, unsort_pl)
    assert sort_pl == [202954, 1627737, 1629673, 203952, 1629672,
                       1627752, 1627747, 1629013, 201145, 1628386]


def test_nine_players(nine_players_period):
    player_id = nba_on_court._players_in_quater(nine_players_period)
    assert (202066 not in player_id) and (len(player_id) == 9)


def test_lost_internet(nine_players_period):
    with pytest.raises(ConnectionError):
        nba_on_court.players_on_court(nine_players_period,
                                      timeout=0.00001)


def test_eleven_players(eleven_players_period):
    player_id = nba_on_court._players_in_quater(eleven_players_period)
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
    df = nba_on_court._fill_columns(ten_players_period, all_id)
    assert df.shape == shape


def test_replace(ten_players_period):
    df = nba_on_court.players_on_court(ten_players_period)
    pl_list = nba_on_court\
        .players_name(df.iloc[0, 34:].reset_index(drop=True))
    assert pl_list == ['Brad Wanamaker', 'Marquese Chriss', 'Jordan Poole',
                       'Andrew Wiggins', 'Eric Paschall', 'Taurean Prince',
                       'Caris LeVert', 'Landry Shamet', 'Jeff Green',
                       'Jarrett Allen']


def test_round_any():
    assert -24.0 == hexdata._round_any(-23.7, 1.5, round)


def test_hex_bounds():
    assert all(np.array([-24.000001,
                         25.500001]) == hexdata._hex_bounds([-23.7, 25.1],
                                                            1.5))


def test_hexbin():
    x = np.arange(5)
    y = np.arange(5)

    xbnds = hexdata._hex_bounds(x, 1.5)
    xbins = np.diff(xbnds)[0] / 1.5
    ybnds = hexdata._hex_bounds(y, 1.5)
    ybins = np.diff(ybnds)[0] / 1.5

    hb = hexdata._hexbin(
        x=x,
        y=y,
        xbins=xbins,
        xbnds=xbnds,
        ybnds=ybnds,
        shape=ybins / xbins
    )

    assert all(hb["cellid"] == np.array([1,  5,  6, 11, 15]))
    assert all(hb["cell"] == np.array([1,  5,  6, 11, 15]))
    assert all(hb["count"] == np.array([1, 1, 1, 1, 1]))
    assert all(hb["xcm"] == np.array([0., 1., 2., 3., 4.]))
    assert all(hb["ycm"] == np.array([0., 1., 2., 3., 4.]))
    assert hb["n"] == 5
    assert all(hb["bnd"] == np.array([4, 4]))
    assert all(hb["dimen"] == np.array([6, 4]))


def test_calculate_hex_coords(players_shot_data):
    df = hexdata.calculate_hex_coords(players_shot_data, np.array([1.5, 1.5]))
    assert df.shape[0] == 48
    assert np.allclose(df.center_x[0], 22.4999)
    assert all(np.unique(df.hexbin_id) == np.array([22, 29, 53,
                                                    68, 291, 309, 312, 438]))


def test_calculate_hexbins_from_shots(players_shot_data, league_average):
    hex_dict = hexdata.calculate_hexbins_from_shots(players_shot_data,
                                                    league_average,
                                                    np.array([1.5, 1.5]))
    assert hex_dict["hex_data"].shape[0] == 48
    assert np.allclose(hex_dict["hex_data"].center_x[0], 22.4999)
    un_hex_id = np.unique(hex_dict["hex_data"].hexbin_id)
    assert all(un_hex_id == np.array([22, 29, 53, 68, 291, 309, 312, 438]))


def test_load_nba_data():
    tmp_folder = Path.cwd().joinpath("load_data")
    tmp_folder.mkdir(parents=True, exist_ok=True)
    nba_on_court.load_nba_data(path=tmp_folder,
                               seasons=2023,
                               data="shotdetail",
                               seasontype="po",
                               league="nba",
                               untar=True)
    assert tmp_folder.joinpath("shotdetail_po_2023.csv").is_file()
    df = pd.read_csv(tmp_folder.joinpath("shotdetail_po_2023.csv"))
    assert df.shape[1] == 24
    tmp_folder.joinpath("shotdetail_po_2023.csv").unlink()
    tmp_folder.rmdir()
