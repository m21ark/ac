# Project AC

## Group Members

| Name          | Number    |
| ------------- | --------- |
| João Alves    | 202007614 |
| Ricardo Matos | 202007962 |
| Marco André   | 202004891 |

## Overview

This project aims at accurately predict the teams that will classify for the next playoff season.

The code was developed in Python 3.10 with Jupyter Notebook.

### Dependencies

The dependencies needed to run the code are present in the `requirements.txt` file. They can be installed using the following command:

```python
pip3 install -r requirements.txt
```

## Relations Attributes Description

### Teams

"year": This field specifies the year associated with the data. It indicates the calendar year to which the data applies or the season year.

"lgID": League Identifier. This field represents the league in which the basketball teams competed. It can include league codes such as NBA (National Basketball Association) or other basketball leagues.

"tmID": Team Identifier. This field contains a unique identifier or code for each basketball team, allowing you to associate data with specific teams.

"franchID": Franchise Identifier. It may represent a unique identifier for the franchise or organization to which the team belongs. Multiple teams may share the same franchise ID if they are part of the same organization.

"confID": Conference Identifier. In basketball leagues with conferences, this field specifies the conference to which the team belongs, such as Eastern Conference or Western Conference.

"divID": Division Identifier. If the league is further divided into divisions, this field indicates the division to which the team is assigned, such as Atlantic Division or Central Division.

"rank": This field represents the team's rank or position within its division or conference during a specific season.

"playoff": Indicates whether the team made the playoffs (e.g., "Y" for Yes, "N" for No). It specifies whether the team qualified for the postseason.

"seeded": Indicates if the team was seeded in the playoffs. A seed indicates the team's position and matchups in the playoff bracket.

"firstRound", "semis", "finals": These fields likely indicate the team's progress in the playoffs. "firstRound" may indicate if the team reached the first round, "semis" if they reached the semifinals, and "finals" if they reached the finals of the playoffs.

"name": This field contains the name or identifier of the basketball team.

Various statistical fields (e.g., "o*fgm," "o_3pm," "d_reb," etc.): These fields represent different statistics related to the team's performance in offense ("o*") and defense ("d\_"). These statistics may include field goals made and attempted, three-pointers made and attempted, rebounds, assists, steals, turnovers, blocks, and points scored.

"tmORB," "tmDRB," "tmTRB": These fields represent team-level statistics for offensive rebounds ("tmORB"), defensive rebounds ("tmDRB"), and total rebounds ("tmTRB").

"opptmORB," "opptmDRB," "opptmTRB": Similar to the previous fields, these represent opponent team-level statistics for offensive rebounds, defensive rebounds, and total rebounds.

"won" and "lost": These fields indicate the total number of games won and lost by the team during the specified season.

"GP": Total Games Played. This field specifies the total number of games played by the team during the season.

"homeW" and "homeL": These fields represent the number of home games won and lost by the team.

"awayW" and "awayL": These fields represent the number of away (road) games won and lost by the team.

"confW" and "confL": These fields represent the number of conference games won and lost by the team.

"min": Minutes played, likely representing the total minutes played by the team during the season.

"attend": Attendance. This field may represent the total attendance (number of spectators) for the team's games during the season.

"arena": Name of the arena or venue where the team played its home games.

## Teams Post

"year",
"tmID",
"lgID",
"W",
"L"

## Series Post

"year": This field represents the year in which the basketball games or series took place. It specifies the calendar year in which the events occurred.

"round": This field indicates the round or stage of the competition or tournament in which the games or series are played. Basketball tournaments often consist of multiple rounds, such as regular season, playoffs, quarterfinals, semifinals, and finals. This field specifies which round is being referred to.

"series": This field likely refers to a specific series of games within the mentioned round. In basketball, a series typically refers to a matchup between two teams where they play a specified number of games, and the team that wins the majority of those games advances in the tournament. For example, in the NBA playoffs, a series is typically a best-of-seven format.

"tmIDWinner": This field contains the identifier or code for the basketball team that won the series. It specifies the team that emerged victorious in the series.

"lgIDWinner": This field contains the league identifier for the winning team. In basketball, there are various basketball leagues around the world, each with its own set of teams. This field identifies the league to which the winning team belongs.

"tmIDLoser": Similar to "tmIDWinner," this field contains the identifier or code for the basketball team that lost the series. It specifies the team that did not advance or win the series.

"lgIDLoser": This field contains the league identifier for the losing team. It identifies the league to which the losing team belongs.

"W": This field represents the number of games won by the team indicated in "tmIDWinner" in the specified series. It indicates the number of games required to win the series. For example, in a best-of-seven series, "W" might have values like 4, 3, 2, or 1, depending on how many games the winning team won.

"L": This field represents the number of games lost by the team indicated in "tmIDWinner" in the specified series. It complements the "W" field and helps determine the outcome of the series. In a best-of-seven series, the "L" value for the winning team is typically lower, often leading to the equation W + L = Total games in the series.

## Players

"bioID": This field likely serves as a unique identifier for each basketball player. It is a code or value that uniquely identifies each player in the dataset.

"pos": This field represents the player's position on the basketball court. In basketball, positions can include point guard (PG), shooting guard (SG), small forward (SF), power forward (PF), and center (C). The "pos" field indicates which of these positions the player primarily played.

"firstseason": This field specifies the first season in which the player participated in professional basketball or entered the league. It indicates the player's rookie season or the beginning of their career in the league.

"lastseason": This field specifies the last season in which the player participated in professional basketball or the league. It indicates when the player retired from professional play.

"height": This field indicates the player's height, typically provided in feet and inches or in centimeters. For example, a player's height might be listed as "6'4" (6 feet 4 inches) or "193 cm."

"weight": This field specifies the player's weight, typically provided in pounds or kilograms. It represents the player's body weight.

"college": This field indicates the college or university where the player played collegiate basketball. It represents the player's alma mater at the college level.

"collegeOther": This field may contain additional information about the player's college career or other colleges they attended. It could include details about transfers or additional educational institutions.

"birthDate": This field specifies the player's date of birth, providing their birthdate.

"deathDate": If applicable, this field would specify the date of death of the player. It would indicate when the player passed away.

## Players Teams

"playerID": This field likely serves as a unique identifier for each basketball player, allowing you to associate player-specific statistics with individual players. It may correspond to the "bioID" field in your "player" dataset.

"year": This field specifies the year or season in which the performance statistics are recorded. It indicates the specific basketball season to which the data applies.

"stint": This field represents the stint or period within a season in which the player played for a particular team. In some cases, players may switch teams within the same season, and this field helps track those changes.

"tmID": This field contains the identifier or code for the basketball team that the player was associated with during the specified season and stint. It indicates the team the player played for.

"lgID": League Identifier. It represents the league in which the player participated during the specified season and stint, such as NBA (National Basketball Association) or another basketball league.

"GP": Games Played. This field indicates the total number of games in which the player participated during the specified season and stint.

"GS": Games Started. It specifies how many games the player started during the specified season and stint. A player can start a game if they are in the starting lineup.

"minutes": This field records the total number of minutes the player played in games during the specified season and stint.

"points": Total points scored by the player during the specified season and stint.

"oRebounds": Offensive Rebounds. It represents the number of times the player secured an offensive rebound during games in the specified season and stint.

"dRebounds": Defensive Rebounds. It represents the number of times the player secured a defensive rebound during games in the specified season and stint.

"rebounds": Total Rebounds. This field indicates the total number of rebounds (offensive + defensive) gathered by the player during games in the specified season and stint.

"assists": Total assists made by the player during games in the specified season and stint.

"steals": Total steals by the player during games in the specified season and stint.

"blocks": Total blocks by the player during games in the specified season and stint.

"turnovers": Total turnovers committed by the player during games in the specified season and stint.

"PF": Personal Fouls. It records the total number of personal fouls committed by the player during games in the specified season and stint.

"fgAttempted": Field Goals Attempted. It represents the total number of field goal attempts made by the player during games in the specified season and stint.

"fgMade": Field Goals Made. This field indicates the total number of successful field goals made by the player during games in the specified season and stint.

"ftAttempted": Free Throws Attempted. It records the total number of free throw attempts by the player during games in the specified season and stint.

"ftMade": Free Throws Made. This field represents the total number of successful free throws made by the player during games in the specified season and stint.

"threeAttempted": Three-Pointers Attempted. It specifies the total number of three-point shot attempts by the player during games in the specified season and stint.

"threeMade": Three-Pointers Made. This field indicates the total number of successful three-point shots made by the player during games in the specified season and stint.

"dq": Disqualified. This field might indicate whether the player was disqualified (ejected or otherwise removed from a game) during any of the games in the specified season and stint.

"PostGP" to "PostDQ": These fields appear to represent similar statistics as the previous fields but for postseason or playoff games, if applicable. They track the player's performance in playoff games, such as games played, points scored, rebounds, etc., during the postseason.

## Coaches

"coachID": This field likely serves as a unique identifier for each basketball coach, allowing you to associate coaching performance statistics with individual coaches.

"year": This field specifies the year or season in which the coaching performance statistics are recorded. It indicates the specific basketball season to which the data applies.

"tmID": This field contains the identifier or code for the basketball team that the coach was associated with during the specified season and stint. It indicates the team the coach coached.

"lgID": League Identifier. It represents the league in which the coach participated during the specified season and stint, such as NBA (National Basketball Association) or another basketball league.

"stint": This field represents the stint or period within a season in which the coach served with a particular team. In some cases, coaches may change teams or have multiple coaching stints within the same season, and this field helps track those changes.

"won": The "won" field indicates the number of games won by the team under the coaching of the specified coach during the specified season and stint.

"lost": The "lost" field indicates the number of games lost by the team under the coaching of the specified coach during the specified season and stint.

"post_wins": This field represents the number of playoff games won by the team under the coaching of the specified coach during the specified season and stint, if applicable. It tracks postseason victories.

"post_losses": This field indicates the number of playoff games lost by the team under the coaching of the specified coach during the specified season and stint, if applicable. It tracks postseason defeats.

## Awards Players

"playerID": This field likely serves as a unique identifier for each basketball player, allowing you to associate specific awards with individual players. It may correspond to the "bioID" or "coachID" fields in other datasets.

"award": This field contains the name or identifier of the award received by the player. Awards in basketball can include MVP (Most Valuable Player), Rookie of the Year, All-Star selections, Defensive Player of the Year, and more.

"year": This field specifies the year in which the award was received by the player. It indicates the calendar year in which the player was recognized for their performance.

"lgID": League Identifier. This field represents the league in which the player received the award. Basketball awards can be league-specific, such as NBA awards for the National Basketball Association or awards from other basketball leagues.
