import warnings
import re
from pathlib import Path
from itertools import product
from urllib.request import urlopen
import tarfile
from functools import lru_cache
from typing import Optional, List, Dict, Union, Sequence
from io import BytesIO, TextIOWrapper
import csv

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv2, commonallplayers
from requests import ConnectionError
from polyleven import levenshtein

warnings.filterwarnings('ignore')


def _convert_timestring_to_second(data: pd.DataFrame, column: str) -> pd.Series:
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


def _players_in_quater(data: pd.DataFrame, all_id: Optional[np.ndarray] = None) -> np.ndarray:
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


def _sort_players(data: pd.DataFrame, all_id: np.ndarray) -> List:
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


def _fill_columns(data: pd.DataFrame, all_id: Union[np.ndarray, List]) -> pd.DataFrame:
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
        data["PCTIMESTRING"] = _convert_timestring_to_second(data, "PCTIMESTRING")
    d = dict()
    for period in data["PERIOD"].unique():
        df = data.loc[data["PERIOD"] == period]
        all_id = _players_in_quater(df)
        if len(all_id) == 10:
            all_id = _sort_players(df, all_id)
            d[period] = _fill_columns(df, all_id)
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
            all_id = _players_in_quater(df, all_id)
            d[period] = _fill_columns(df, all_id)

    return pd.concat(d, axis=0, ignore_index=True).rename(columns={
        "PLAYER1": 'AWAY_PLAYER1',
        "PLAYER2": 'AWAY_PLAYER2',
        "PLAYER3": 'AWAY_PLAYER3',
        "PLAYER4": 'AWAY_PLAYER4',
        "PLAYER5": 'AWAY_PLAYER5',
        "PLAYER6": 'HOME_PLAYER1',
        "PLAYER7": "HOME_PLAYER2",
        "PLAYER8": "HOME_PLAYER3",
        "PLAYER9": "HOME_PLAYER4",
        "PLAYER10": "HOME_PLAYER5"
    })


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
        season=args.get("season", "2022-23"),
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


def load_nba_data(path: Union[Path, str] = Path.cwd(),
                  seasons: Union[Sequence, int] = range(1996, 2023),
                  data: Union[Sequence, str] = ("datanba", "nbastats", "pbpstats",
                                                "shotdetail", "cdnnba", "nbastatsv3"),
                  seasontype: str = 'rg',
                  league: str = 'nba',
                  untar: bool = False,
                  in_memory: bool = False,
                  use_pandas: bool = True) -> Optional[Union[List, pd.DataFrame]]:
    """
    Loading a nba play-by-play dataset from github repository https://github.com/shufinskiy/nba_data

    Args:
        path (Union[Path, str]): Path where downloaded file should be saved on the hard disk
        seasons (Union[Sequence, int]): Sequence or integer of the year of start of season
        data (Union[Sequence, str]): Sequence or string of data types to load
        seasontype (str): Part of season: rg - Regular Season, po - Playoffs
        league (str): Name league: NBA or WNBA
        untar (bool): Logical: do need to untar loaded archive
        in_memory (bool): Logical: If True dataset is loaded into workflow, without saving file to disk
        use_pandas (bool): Logical: If True dataset is loaded how pd.DataFrame, else List[List[str]]. Ignore if in_memory=False

    Returns:
        Optional[pd.DataFrame, List]: If in_memory=True and use_pandas=True return dataset how pd.DataFrame.
        If use_pandas=False return dataset how List[List[str]]
        If in_memory=False return None
    """
    if isinstance(path, str):
        path = Path(path)
    if isinstance(seasons, int):
        seasons = (seasons,)
    if isinstance(data, str):
        data = (data,)

    if (len(data) > 1) & in_memory:
        raise ValueError("Parameter in_memory=True available only when loading a single data type")

    if seasontype == 'rg':
        need_data = tuple(["_".join([data, str(season)]) for (data, season) in product(data, seasons)])
    elif seasontype == 'po':
        need_data = tuple(["_".join([data, seasontype, str(season)]) \
                           for (data, seasontype, season) in product(data, (seasontype,), seasons)])
    else:
        need_data_rg = tuple(["_".join([data, str(season)]) for (data, season) in product(data, seasons)])
        need_data_po = tuple(["_".join([data, seasontype, str(season)]) \
                              for (data, seasontype, season) in product(data, ('po',), seasons)])
        need_data = need_data_rg + need_data_po
    if league.lower() == 'wnba':
        need_data = ['wnba_' + x for x in need_data]

    check_data = [file + ".csv" if untar else "tar.xz" for file in need_data]
    not_exists = [not path.joinpath(check_file).is_file() for check_file in check_data]

    need_data = [file for (file, not_exist) in zip(need_data, not_exists) if not_exist]

    with urlopen("https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt") as f:
        v = f.read().decode('utf-8').strip()

    name_v = [string.split("=")[0] for string in v.split("\n")]
    element_v = [string.split("=")[1] for string in v.split("\n")]

    need_name = [name for name in name_v if name in need_data]
    need_element = [element for (name, element) in zip(name_v, element_v) if name in need_data]

    if in_memory:
        if use_pandas:
            table = pd.DataFrame()
        else:
            table = []
    for i in range(len(need_name)):
        with urlopen(need_element[i]) as response:
            if response.status != 200:
                raise Exception(f"Failed to download file: {response.status}")
            file_content = response.read()
            if in_memory:
                with tarfile.open(fileobj=BytesIO(file_content), mode='r:xz') as tar:
                    csv_file_name = "".join([need_name[i], ".csv"])
                    csv_file = tar.extractfile(csv_file_name)
                    if use_pandas:
                        table = pd.concat([table, pd.read_csv(csv_file)], axis=0, ignore_index=True)
                    else:
                        csv_reader = csv.reader(TextIOWrapper(csv_file, encoding="utf-8"))
                        for row in csv_reader:
                            table.append(row)
            else:
                with path.joinpath("".join([need_name[i], ".tar.xz"])).open(mode='wb') as f:
                    f.write(file_content)
                if untar:
                    with tarfile.open(path.joinpath("".join([need_name[i], ".tar.xz"]))) as f:
                        f.extract("".join([need_name[i], ".csv"]), path)

                    path.joinpath("".join([need_name[i], ".tar.xz"])).unlink()
    if in_memory:
        return table
    else:
        return None


def _concat_description(homedescription: pd.Series,
                        neutraldescription: pd.Series,
                        visitordescription: pd.Series) -> pd.Series:
    """
    Merge description columns in nbastats play-by-play-data

    Args:
        homedescription (pd.Series): event description for home team
        neutraldescription (pd.Series): neutral event description
        visitordescription (pd.Series): event description for away team

    Returns:
        pd.Series: columns with merge all description
    """
    return pd.Series([re.sub(r' +', r' ', ' '.join([home, neutral, visit]).strip()) for home, neutral, visit \
                      in zip((homedescription).where(~pd.isna(homedescription), ''),
                             (neutraldescription).where(~pd.isna(neutraldescription), ''),
                             (visitordescription).where(~pd.isna(visitordescription), ''))])


def _transform_nbastats(nbastats: pd.DataFrame) -> pd.DataFrame:
    """
    Convert timestring column to second and merge description columns.

    Args:
        nbastats (pd.DataFrame): nbastats pla-by-play dataframe

    Returns:
        pd.DataFrame: nbastats play-by-play dataframe with update columns
    """
    nbastats['PCTIMESTRING'] = _convert_timestring_to_second(nbastats, 'PCTIMESTRING')
    nbastats['DESCRIPTION'] = _concat_description(nbastats.HOMEDESCRIPTION,
                                                  nbastats.NEUTRALDESCRIPTION,
                                                  nbastats.VISITORDESCRIPTION).str.lower()
    nbastats.drop(columns=['HOMEDESCRIPTION', 'NEUTRALDESCRIPTION', 'VISITORDESCRIPTION'], inplace=True)
    return nbastats


def _transform_pbpstats(pbpstats: pd.DataFrame) -> pd.DataFrame:
    """
    Data transformation pbpstats.com to merge with nbastats data

    Args:
        pbpstats (pd.DataFrame): pbpstats.com play-by-play dataframe

    Returns:
        pd.DataFrame: pbpstats.com play-by-play dataframe with update columns
    """
    pbpstats['ENDTIME'] = _convert_timestring_to_second(pbpstats, 'ENDTIME')
    pbpstats['STARTTIME'] = _convert_timestring_to_second(pbpstats, 'STARTTIME')
    pbpstats['EVENT_IN_POSS'] = pbpstats.groupby('ENDTIME').cumcount()
    pbpstats = pbpstats.sort_values(by=['STARTTIME', 'ENDTIME', 'EVENT_IN_POSS']).reset_index(drop=True)
    pbpstats['DESCRIPTION'] = pd.Series(['' if pd.isna(desc) else desc for desc in pbpstats['DESCRIPTION']])
    pbpstats['DESCRIPTION'] = pd.Series([re.sub(' +', ' ', x) for x in pbpstats['DESCRIPTION']]).str.lower()
    return pbpstats


def left_join_nbastats(nbastats: pd.DataFrame, pbpstats: pd.DataFrame, alpha: int = 5,
                       beta: float = 0.2, debug: bool = False,
                       warnings: bool = False) -> Union[pd.DataFrame, np.ndarray]:
    """
    Left join pbpstats.com play-by-play data to nbastats play-by-play data

    Args:
        nbastats (pd.DataFrame): nbastats play-by-play dataframe
        pbpstats (pd.DataFrame): pbpstats.com play-by-play dataframe
        alpha (int): time range in sec to merge data
        beta (float): max Levenstein distance for merge
        debug (bool): enable debug mode
        warnings (bool): print information about error in merge or not

    Returns:
        Union[pd.DataFrame, np.ndarray]: merge play-by-play dataframe
    """
    verbose_warnings = False

    nbastats = _transform_nbastats(nbastats.reset_index(drop=True))
    pbpstats = _transform_pbpstats(pbpstats.reset_index(drop=True))

    cnt_period = np.max(nbastats['PERIOD'])
    df = pd.DataFrame()
    if debug:
        debug_ind = 0
        db_array = np.zeros(pbpstats.shape[0])
    for nperiod in range(1, cnt_period + 1):
        nba_period = nbastats[nbastats['PERIOD'] == nperiod].reset_index(drop=True)
        pbp_period = pbpstats[pbpstats['PERIOD'] == nperiod].reset_index(drop=True)

        nba_period['STATS_KEY'] = nba_period['GAME_ID'].index
        pbp_period['PBP_KEY'] = pbp_period['GAMEID'].index
        pbp_period['STATS_KEY'] = 0

        for i in range(nba_period.shape[0]):
            time = nba_period['PCTIMESTRING'].iloc[i]
            nba_desc = nba_period['DESCRIPTION'].iloc[i]

            tmp_pbp = pbp_period.loc[(pbp_period['STARTTIME'] < time + alpha) & (pbp_period['ENDTIME'] > time - alpha),
            ['DESCRIPTION', 'PBP_KEY']]
            pbp_desc = tmp_pbp['DESCRIPTION'].reset_index(drop=True)
            pbp_key = tmp_pbp['PBP_KEY'].reset_index(drop=True)

            dist_lev = np.array([levenshtein(nba_desc, pbp_desc[i]) / np.max([len(nba_desc), len(pbp_desc[i])]) \
                                 for i, _ in enumerate(pbp_desc)])

            ind = np.where(dist_lev < beta)[0]
            if len(ind) == 0:
                continue
            elif len(ind) > 1:
                ind_cycle = 0
                for _ in range(len(ind)):
                    if pbp_period.loc[pbp_period['PBP_KEY'] == pbp_key[ind[ind_cycle]], 'STATS_KEY'] \
                            .reset_index(drop=True).iloc[0] == 0:
                        ind = ind[ind_cycle]
                        break
                    ind_cycle += 1
            try:
                ind = ind[0]
            except IndexError:
                pass
            pbp_period.loc[pbp_period['PBP_KEY'] == pbp_key[ind], 'STATS_KEY'] = nba_period['STATS_KEY'].iloc[i]
        try:
            assert len(pbp_period.STATS_KEY) == len(pbp_period.STATS_KEY.unique())
        except AssertionError:
            if warnings:
                verbose_warning = True
        if debug:
            n = pbp_period.STATS_KEY.to_numpy()
            db_array[debug_ind:debug_ind + len(n)] = n
            debug_ind += len(n)
        else:
            df = pd.concat([df,
                            (nba_period
                             .merge(pbp_period.drop(columns=['PERIOD']), how='left',
                                    on='STATS_KEY', suffixes=['_STATS', '_PBP'])
                             .drop(columns=['STATS_KEY', 'PBP_KEY', 'EVENT_IN_POSS']))],
                           axis=0, ignore_index=True)
    if debug:
        if verbose_warnings:
            print('Warning: there may be an error in data')
        return db_array
    else:
        if verbose_warnings:
            print('Warning: there may be an error in data')
        return df


def left_join_pbpstats(nbastats: pd.DataFrame, pbpstats: pd.DataFrame, alpha: int = 5,
                       beta: float = 0.2, debug: bool = False,
                       warnings: bool = False) -> Union[pd.DataFrame, np.ndarray]:
    """
    Left join nbastats play-by-play data to pbpstats.com play-by-play data

    Args:
        nbastats (pd.DataFrame): nbastats play-by-play dataframe
        pbpstats (pd.DataFrame): pbpstats.com play-by-play dataframe
        alpha (int): time range in sec to merge data
        beta (float): max Levenstein distance for merge
        debug (bool): enable debug mode
        warnings (bool): print information about error in merge or not

    Returns:
        Union[pd.DataFrame, np.ndarray]: merge play-by-play dataframe
    """
    verbose_warnings = False

    nbastats = _transform_nbastats(nbastats.reset_index(drop=True))
    pbpstats = _transform_pbpstats(pbpstats.reset_index(drop=True))

    cnt_period = np.max(nbastats['PERIOD'])
    df = pd.DataFrame()
    if debug:
        debug_ind = 0
        db_array = np.zeros(pbpstats.shape[0])
    for nperiod in range(1, cnt_period + 1):
        nba_period = nbastats[nbastats['PERIOD'] == nperiod].reset_index(drop=True)
        pbp_period = pbpstats[pbpstats['PERIOD'] == nperiod].reset_index(drop=True)

        nba_period['STATS_KEY'] = nba_period['GAME_ID'].index + 1
        pbp_period['PBP_KEY'] = pbp_period['GAMEID'].index
        pbp_period['STATS_KEY'] = 1000

        for i in range(pbp_period.shape[0]):
            time = np.array([pbp_period['STARTTIME'].iloc[i], pbp_period['ENDTIME'].iloc[i]])
            pbp_desc = pbp_period['DESCRIPTION'].iloc[i]

            tmp_nba = nba_period.loc[(nba_period['PCTIMESTRING'] > time[0] - alpha) \
                                     & (nba_period['PCTIMESTRING'] < time[1] + alpha),
            ['DESCRIPTION', 'STATS_KEY']]
            nba_desc = tmp_nba['DESCRIPTION'].reset_index(drop=True)
            nba_key = tmp_nba['STATS_KEY'].reset_index(drop=True)

            dist_lev = np.array([levenshtein(pbp_desc, nba_desc[i]) / np.max([len(pbp_desc), len(nba_desc[i])]) \
                                 for i, _ in enumerate(nba_desc)])

            ind = np.where(dist_lev < beta)[0]
            if len(ind) == 0:
                continue
            elif len(ind) > 1:
                ind_cycle = 0
                for _ in range(len(ind)):
                    if pbp_period.loc[pbp_period['STATS_KEY'] == nba_key[ind[ind_cycle]]].shape[0] == 0:
                        ind = ind[ind_cycle]
                        break
                    ind_cycle += 1
            try:
                ind = ind[0]
            except IndexError:
                pass
            pbp_period.loc[pbp_period['PBP_KEY'] == i,
            'STATS_KEY'] = nba_period.loc[nba_period['STATS_KEY'] == nba_key[ind],
            'STATS_KEY'].reset_index(drop=True).iloc[0]

        try:
            assert len(pbp_period.STATS_KEY) == len(pbp_period.STATS_KEY.unique())
        except AssertionError:
            if warnings:
                verbose_warnings = True
        if debug:
            n = pbp_period.STATS_KEY.to_numpy()
            db_array[debug_ind:debug_ind + len(n)] = n
            debug_ind += len(n)
        else:
            df = pd.concat([df,
                            (pbp_period
                             .merge(nba_period.drop(columns=['PERIOD']), how='left',
                                    on='STATS_KEY', suffixes=['_PBP', '_STATS'], sort=True)
                             .drop(columns=['STATS_KEY', 'PBP_KEY', 'EVENT_IN_POSS']))],
                           axis=0, ignore_index=True)
    if debug:
        if verbose_warnings:
            print('Warning: there may be an error in data')
        return db_array
    else:
        if verbose_warnings:
            print('Warning: there may be an error in data')
        return df
