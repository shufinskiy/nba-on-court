[![PyPI](https://img.shields.io/pypi/v/nba-on-court)](https://pypi.python.org/pypi/nba-on-court)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/shufinskiy/nba-on-court/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/badge/nba-on-court)](https://pepy.tech/project/nba-on-court)
[![Telegram](https://img.shields.io/badge/telegram-write%20me-blue.svg)](https://t.me/brains14482)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MuUcSj59kl-FO4X-LBRxxOZtEZLfWLDT)

Fast download of play-by-play data and adding data about players on court in NBA games.
================================================

Update [31-05-2024]: Added the opportunity to work with WNBA data from the repository [nba_data](https://github.com/shufinskiy/nba_data)
------

**nba_on_court** package allows you next things:
1. Fast download play-by-play data from [nba_data](https://github.com/shufinskiy/nba_data) repository
2. Add to play-by-play data information about players who were on court at any given time.
3. Merge play-by-play data from different sources

Instalation
-----------

```bash
pip install nba-on-court
```

Tutorial
--------
To understand work of library, you can study tutorials: in [russian](https://github.com/shufinskiy/nba-on-court/blob/main/example/tutorial_ru.ipynb) and [english](https://github.com/shufinskiy/nba-on-court/blob/main/example/tutorial_en.ipynb). There is also an interactive tutorial on **Google Colab**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MuUcSj59kl-FO4X-LBRxxOZtEZLfWLDT)

### Fast download play-by-play data from [nba_data](https://github.com/shufinskiy/nba_data) repository

With help of the previous version of the library, it was not possible to get play-by-play data, for this it was 
necessary to use third-party solutions, for example the library [nba_api](https://github.com/swar/nba_api). The disadvantage of this approach is speed: 
NBA website has quite strict limits on the number of requests, so collecting play-by-play data from one 
season can take several hours.

[nba_data](https://github.com/shufinskiy/nba_data) repository, which containing play-by-play data from three sources (nba.stats.com , pbpstats.com , data.nba.com),
as well as shotdetail for all games (regular season and playoffs) since the 1996/97 season 
(data from pbpstats.com and data.nba.com from the season of their appearance). 
Due to the fact that you just download a file from github, downloading one season of play-by-play data will take several 
seconds (depends on your internet speed). In 5-10 minutes, you can download the entire array of data for 28 seasons. 
Fast loading of play-by-play data is carried out using the **load_nba_data** function.

```python
import nba_on_court as noc
noc.load_nba_data(seasons=2022, data='nbastats')
```

### Add to play-by-play data information about players who were on court at any given time

Play-by-play NBA data contains information about each event in the game
(throw, substitution, foul, etc.) and players who participated in it
(PLAYER1_ID, PLAYER2_ID, PLAYER3_ID).

From this data, we get a list of players who were on court in this
quarter. Then, we need to filter this list to 10 people who started
quarter. This is done by analyzing substitutions in quarter.

**players_on_court** takes play-by-play data as input and returns it with 10
columns of the PLAYER_ID of players who were on court at each time.

**players_name** allows you to replace PLAYER_ID with first and last name of player.
This allows user to understand exactly which players were on court (few know PLAYER_ID
all players in NBA),but it is not necessary to do this before calculations, because the
player's NAME_SURNAME is not unique, unlike PLAYER_ID.

```python


import nba_on_court as noc
from nba_api.stats.endpoints import playbyplayv2

pbp = playbyplayv2.PlayByPlayV2(game_id="0022100001").play_by_play.get_data_frame()
pbp_with_players = noc.players_on_court(pbp)
len(pbp_with_players.columns) - len(pbp.columns)
10

players_id = list(pbp_with_players.iloc[0, 34:].reset_index(drop=True))
print(players_id)
[201142, 1629651, 201933, 201935, 203925, 201572, 201950, 1628960, 203114, 203507]

players_name = noc.players_name(players_id)
print(players_name)
['Kevin Durant', 'Nic Claxton', 'Blake Griffin', 'James Harden', 'Joe Harris',
 'Brook Lopez', 'Jrue Holiday', 'Grayson Allen', 'Khris Middleton', 'Giannis Antetokounmpo']
```
You can also replace the PLAYER_ID with the player's name in the entire data frame at once.

```python
    cols = ["PLAYER1", "PLAYER2", "PLAYER3", "PLAYER4", "PLAYER5", "PLAYER6", "PLAYER7", "PLAYER8", "PLAYER9", "PLAYER10"]
    pbp_with_players.loc[:, cols] = pbp_with_players.loc[:, cols].apply(noc.players_name, result_type="expand")
```

### Merge play-by-play data from different sources

Sometimes you need to combine data from different sources to solve a problem. For example, we want to find out how the 
on/off of a partner on the floor affects the player's shot selection. To do this, we need detailed throw data (where 
there are coordinates and throw zones), as well as play-by-play data with information about the presence on court in 
order to divide the throws according to condition. In the repository nba_data 3 data sources (nba.stats, data.stats and 
shotdetail) have a single source: NBA website. Therefore, it is quite easy to combine them by two keys (the name of the 
columns differs in different sources):

- Game ID
- Event ID

With data from pbpstats.com more complicated: they initially have a another structure (grouped by possessions), 
so they do not have an event ID. At the same time, they contain useful information that is not explicitly available 
in other sources (time of possessions, type of possessions start, url of video episode). The only way to combine them 
is to use the event DESCRIPTION. The problem here is that the descriptions in nba.stats and pbpstats also do not match 
and an attempt to merge them directly will lead to the loss of a certain number of rows.

To solve this problem, I created the functions **left_join_nbastats** and **left_join_pbpstats**. They allow you to 
merge play-by-play data with NBA.stats and pbpstats with almost no errors.

```python
import pandas as pd
import nba_on_court as noc

noc.load_nba_data(seasons=2022, data=('nbastats', 'pbpstats'), seasontype='po', untar=True)

nbastats = pd.read_csv('nbastats_po_2022.csv')
pbpstats = pd.read_csv('pbpstats_po_2022.csv')

nbastats = nbastats.loc[nbastats['GAME_ID'] == 42200405].reset_index(drop=True)
pbpstats = pbpstats.loc[pbpstats['GAMEID'] == 42200405].reset_index(drop=True)

print(nbastats.shape, pbpstats.shape)
((463, 34), (396, 19))

full_pbp = noc.left_join_nbastats(nbastats, pbpstats)
print(full_pbp.shape)
(463, 50)
```

### Contact me:

If you have questions or proposal about dataset, you can write me convenient for you in a way.

<div id="header" align="left">
  <div id="badges">
    <a href="https://www.linkedin.com/in/vladislav-shufinskiy/">
      <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn Badge"/>
    </a>
    <a href="https://t.me/brains14482">
      <img src="https://img.shields.io/badge/Telegram-blue?style=for-the-badge&logo=telegram&logoColor=white" alt="Telegram Badge"/>
    </a>
    <a href="https://twitter.com/vshufinskiy">
      <img src="https://img.shields.io/badge/Twitter-blue?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter Badge"/>
    </a>
  </div>
</div>
