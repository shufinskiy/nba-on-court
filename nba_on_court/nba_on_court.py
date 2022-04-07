import warnings
from functools import lru_cache
from typing import Optional, List, Dict, Union

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv2, commonallplayers
from requests import ConnectionError

warnings.filterwarnings('ignore')


def convert_timestring_to_second(data: pd.DataFrame, column: str) -> pd.Series:
    """Converting a string of game time into a numeric of seconds from the start of the game.

    Args:
        data (pd.DataFrame): play-by-play data from stats.nba.com
        column (str): name of column with game time

    Returns:
        pd.Series: seconds from start of game
    """
    data.columns = [x.upper() for x in data.columns]
    s = data.loc[:, column].str.split(pat=":", expand=True).astype('int32')
    s.columns = ["MIN", "SEC"]
    timegame = pd.Series([np.abs((minute * 60 + sec) - 720 * period) if period < 5 \
                              else np.abs((minute * 60 + sec) - (2880 + 300 * (period - 4))) \
                          for (minute, sec, period) in zip(s["MIN"], s["SEC"], data["PERIOD"])])
    return timegame


def players_in_quater(data: pd.DataFrame, all_id: Optional[np.ndarray] = None) -> np.ndarray:
    """Getting array of PLAYER_ID players who were on court at start of quarter.

    Args:
        data (pd.Dataframe): play-by-play data from stats.nba.com
        all_id (np.array): Array of PLAYER_ID of all players who appeared on court in quarter from nba boxscore.
        Used only when 10 players cannot be retrieved from play-by-play data. By default, None.

    Returns:
        np.array: PLAYER_ID of players on court at beginning of quarter.
    """
    data.columns = [x.upper() for x in data.columns]
    if all_id is None:
        pl1_id = data.loc[(~data["EVENTMSGTYPE"].isin([9, 18])) & (~data["PERSON1TYPE"].isin([6, 7])) & \
                          (~pd.isna(data["PLAYER1_NAME"])), "PLAYER1_ID"].unique()
        pl2_id = data.loc[(~data["EVENTMSGTYPE"].isin([9, 18])) & (~data["PERSON2TYPE"].isin([6, 7])) & \
                          (~pd.isna(data["PLAYER2_NAME"])), "PLAYER1_ID"].unique()
        pl3_id = data.loc[(~data["EVENTMSGTYPE"].isin([9, 18])) & (~data["PERSON3TYPE"].isin([6, 7])) & \
                          (~pd.isna(data["PLAYER3_NAME"])), "PLAYER3_ID"].unique()
        all_id = np.unique(np.concatenate((pl1_id, pl2_id, pl3_id)))
        all_id = all_id[(all_id != 0) & (all_id < 1610612737)]

    sub_off = data.loc[data["EVENTMSGTYPE"] == 8, "PLAYER1_ID"].unique()
    sub_on = data.loc[data["EVENTMSGTYPE"] == 8, "PLAYER2_ID"].unique()
    all_id = all_id[~np.in1d(all_id, sub_on[~np.in1d(sub_on, sub_off)])]
    sub_on_off = sub_on[np.in1d(sub_on, sub_off)]

    for i in sub_on_off:
        on = np.min(data.loc[(data["EVENTMSGTYPE"] == 8) & (data["PLAYER2_ID"] == i), "PCTIMESTRING"])
        off = np.min(data.loc[(data["EVENTMSGTYPE"] == 8) & (data["PLAYER1_ID"] == i), "PCTIMESTRING"])
        if off > on:
            all_id = all_id[~np.in1d(all_id, i)]
        elif off == on:
            on_event = np.min(data.loc[(data["EVENTMSGTYPE"] == 8) & (data["PLAYER2_ID"] == i), "EVENTNUM"])
            off_event = np.min(data.loc[(data["EVENTMSGTYPE"] == 8) & (data["PLAYER1_ID"] == i), "EVENTNUM"])
            if off_event > on_event:
                all_id = all_id[~np.in1d(all_id, i)]
    return all_id


def sort_players(data: pd.DataFrame, all_id: np.ndarray) -> List:
    """Sorting players on court by teams (first away team, then home team)

    Args:
        data (pd.Dataframe): play-by-play data from stats.nba.com
        all_id: unsorted by teams array of PLAYER_ID

    Returns:
`       list: sorted by teams (first away team, then home team) array of PLAYER_ID of players
        on court at beginning of quarter.
    """
    data.columns = [x.upper() for x in data.columns]
    pl1_id = data.loc[(data["PLAYER1_ID"] != 0) & (data["PERSON1TYPE"].isin([4, 5])), ["PLAYER1_ID", "PERSON1TYPE"]] \
        .rename(columns={"PLAYER1_ID": "PLAYER_ID", "PERSON1TYPE": "PERSONTYPE"})
    pl2_id = data.loc[(data["PLAYER2_ID"] != 0) & (data["PERSON2TYPE"].isin([4, 5])), ["PLAYER2_ID", "PERSON2TYPE"]] \
        .rename(columns={"PLAYER2_ID": "PLAYER_ID", "PERSON2TYPE": "PERSONTYPE"})
    pl3_id = data.loc[(data["PLAYER3_ID"] != 0) & (data["PERSON3TYPE"].isin([4, 5])), ["PLAYER3_ID", "PERSON3TYPE"]] \
        .rename(columns={"PLAYER3_ID": "PLAYER_ID", "PERSON3TYPE": "PERSONTYPE"})
    pl_id = np.array(pd.concat([pl1_id, pl2_id, pl3_id], axis=0, ignore_index=True).drop_duplicates() \
                     .sort_values(by="PERSONTYPE", ascending=False).loc[:, "PLAYER_ID"])
    pl_id = [x for x in pl_id if x in all_id]
    return pl_id


def fill_columns(data: pd.DataFrame, all_id: Union[np.ndarray, List]) -> pd.DataFrame:
    """Adding columns with PLAYER_ID on court

    Args:
        data (pd.DataFrame): play-by-play data from stats.nba.com
        all_id (np.array): sorted by teams (before away team, after home team) array of PLAYER_ID of players
        on court at beginning of quarter.

    Returns:
        pd.Dataframe: play-by-play data with PLAYER_ID on court
    """
    data.columns = [x.upper() for x in data.columns]
    for i in range(1, 11):
        data.loc[:, "PLAYER" + str(i)] = all_id[i - 1]
        subs = data.loc[(data["EVENTMSGTYPE"] == 8) & (data["PLAYER1_ID"] == data["PLAYER" + str(i)])]
        nsub = subs.shape[0]
        while nsub > 0:
            n = subs.index[0]
            player_on = subs.iloc[0]["PLAYER2_ID"]
            data.loc[n:, "PLAYER" + str(i)] = player_on
            subs = data.loc[(data["EVENTMSGTYPE"] == 8) & (data["PLAYER1_ID"] == data["PLAYER" + str(i)])]
            nsub = subs.shape[0]
    return data


def players_on_court(data: pd.DataFrame, **kwargs: Dict[str, float]) -> pd.DataFrame:
    """Adding players on court to play-by-play data from stats.nba.com

    Args:
        data (pd.DataFrame): play-by-play data from stats.nba.com
        **kwargs (Dict): Arbitrary keyword arguments.

    Returns:
        pd.DataFrame: play-by-play data with players on court
    """
    args = kwargs
    data.columns = [x.upper() for x in data.columns]
    if isinstance(data["PCTIMESTRING"][0], str):
        data["PCTIMESTRING"] = convert_timestring_to_second(data, "PCTIMESTRING")
    d = dict()
    for period in data["PERIOD"].unique():
        df = data.loc[data["PERIOD"] == period]
        all_id = players_in_quater(df)
        if len(all_id) == 10:
            all_id = sort_players(df, all_id)
            d[period] = fill_columns(df, all_id)
        else:
            retry = 0
            bx = ""
            while retry < args.get("retry", 5):
                try:
                    bx = boxscoretraditionalv2.BoxScoreTraditionalV2(
                        game_id=args.get("game_id", "00" + str(df.GAME_ID.unique()[0])),
                        start_period=args.get("start_period", str(period)),
                        end_period=args.get("end_period", str(period)),
                        range_type=args.get("range_type", "1"),
                        timeout=args.get("timeout", 10)
                    )
                    break
                except ConnectionError as e:
                    bx = e
                    retry += 1
            if not isinstance(bx, boxscoretraditionalv2.BoxScoreTraditionalV2):
                raise ConnectionError(bx)
            player_stats = bx.player_stats.get_data_frame()
            all_id = np.array([player_stats.PLAYER_ID]).ravel()
            all_id = players_in_quater(df, all_id)
            d[period] = fill_columns(df, all_id)

    return pd.concat(d, axis=0, ignore_index=True)


@lru_cache()
def _cache_player_info(**kwargs: Dict[str, float]) -> commonallplayers.CommonAllPlayers:
    """Getting data on all players in history of NBA

    Args:
        **kwargs (Dict): Arbitrary keyword arguments.

    Returns:
        commonallplayers.CommonAllPlayers: class nba_api with information about all players in NBA history
    """
    args = kwargs
    info = commonallplayers.CommonAllPlayers(
        is_only_current_season=args.get("is_only_current_season", 0),
        league_id=args.get("league_id", "00"),
        season=args.get("season", "2021-22"),
        timeout=args.get("timeout", 10)
    )
    return info


def players_name(id_players: Union[pd.Series, np.ndarray, List],
                       player_data: Optional[pd.DataFrame] = None, **kwargs: Dict[str, float]) -> List:
    """Replacing player's PLAYER_ID with his full name

    Args:
        id_players: List of player IDs on court
        player_data: Data on all players in history of NBA. If None, data is requested from stats.nba.com.
            By default, None.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        list: List of names of players on court
    """
    args = kwargs
    if player_data is None:
        retry = 0
        info = ""
        while retry < args.get("retry", 5):
            try:
                info = _cache_player_info()
                break
            except ConnectionError as e:
                info = e
                retry += 1
        if not isinstance(info, commonallplayers.CommonAllPlayers):
            raise ConnectionError(info)
        player_data = info.common_all_players.get_data_frame()

    player_data.columns = [x.upper() for x in player_data.columns]
    name_players = []
    for player in id_players:
        pl_id = player_data.loc[player_data["PERSON_ID"] == player, "DISPLAY_FIRST_LAST"].reset_index(drop=True)[0]
        name_players.append(pl_id)

    return name_players
