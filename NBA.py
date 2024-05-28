import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import json


def generate_all_nba_json(first_team, second_team, third_team, first_rookie_team, second_rookie_team, filename):
    data = {
        "first all-nba team": first_team,
        "second all-nba team": second_team,
        "third all-nba team": third_team,
        "first rookie all-nba team": first_rookie_team,
        "second rookie all-nba team": second_rookie_team
    }

    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=2)



def remove_duplicate_players(first_team, second_team, third_team):
    all_players = {}
    for player, frequency in first_team.items():
        all_players[player] = max(all_players.get(player, 0), frequency)
    for player, frequency in second_team.items():
        all_players[player] = max(all_players.get(player, 0), frequency)
    for player, frequency in third_team.items():
        all_players[player] = max(all_players.get(player, 0), frequency)

    def filter_team(team):
        return {player: frequency for player, frequency in team.items() if frequency == all_players[player]}

    filtered_first_team = filter_team(first_team)
    filtered_second_team = filter_team(second_team)
    filtered_third_team = filter_team(third_team)

    return filtered_first_team, filtered_second_team, filtered_third_team

def filter_top_players_by_year(data, stats, percentage):
    top_players_per_year = []
    start_year = data['YEAR'].min()
    end_year = data['YEAR'].max()
    for year in range(start_year, end_year + 1):
        filtered_data = data[data['YEAR'] == year].copy()
        if not filtered_data.empty:
            filtered_data['Average_Stats'] = filtered_data[stats].mean(axis=1)
            sorted_data = filtered_data.sort_values(by='Average_Stats', ascending=False).copy()
            top_percentage = int(len(sorted_data) * percentage)
            top_players = sorted_data.head(top_percentage).copy()
            top_players_per_year.append(top_players)
    top_players_all_years = pd.concat(top_players_per_year).reset_index(drop=True)
    return top_players_all_years



def corrlation_cal(data, features, year, value):
    data_00_22 = data[data['YEAR'] != year].copy()
    correlation_matrix = data_00_22[features + ['Result']].corr()
    sorted_correlation = correlation_matrix['Result'].drop('Result').sort_values(ascending=False)

    significant_features = sorted_correlation[sorted_correlation >= value].index.tolist()
    return significant_features


def filter_G_position(data):
    return data[data['Position'].str.contains('G') | (data['Position'] == 'NONE')]


def filter_C_position(data):
    return data[data['Position'].str.contains('C') | (data['Position'] == 'NONE')]


def filter_F_position(data):
    return data[data['Position'].str.contains('F') | (data['Position'] == 'NONE')]


def evaluate(models, BIG_DATA):
    total_player_count_100 = {}
    total_player_count_60 = {}
    total_player_count_20 = {}

    for model in models:
        for data in BIG_DATA:
            X_train, y_train, X_test, y_test, player_data = data
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            player_count_100 = {}  # first team
            player_count_60 = {}  # second team
            player_count_20 = {}  # third team

            for player_name, pred_label in zip(player_data['PLAYER_NAME'], y_pred):
                if pred_label == 3:
                    player_count_100[player_name] = player_count_100.get(player_name, 0) + 1
                if pred_label == 2:
                    player_count_60[player_name] = player_count_60.get(player_name, 0) + 1
                if pred_label == 1:
                    player_count_20[player_name] = player_count_20.get(player_name, 0) + 1

            for player_name, count in player_count_100.items():
                total_player_count_100[player_name] = total_player_count_100.get(player_name, 0) + count
            for player_name, count in player_count_60.items():
                total_player_count_60[player_name] = total_player_count_60.get(player_name, 0) + count
            for player_name, count in player_count_20.items():
                total_player_count_20[player_name] = total_player_count_20.get(player_name, 0) + count

    total_player_count_100 = {k: v for k, v in
                              sorted(total_player_count_100.items(), key=lambda item: item[1], reverse=True)}
    total_player_count_60 = {k: v for k, v in
                             sorted(total_player_count_60.items(), key=lambda item: item[1], reverse=True)}
    total_player_count_20 = {k: v for k, v in
                             sorted(total_player_count_20.items(), key=lambda item: item[1], reverse=True)}



    return total_player_count_100, total_player_count_60, total_player_count_20


def make_collection(data, year, target, features):
    train_data = data[data['YEAR'] != year].copy()
    test_data = data[data['YEAR'] == year].copy()
    player_data = data[data['YEAR'] == year].copy()
    columns_to_keep = features + [target]

    train_data.drop(columns=[col for col in train_data.columns if col not in columns_to_keep], inplace=True)
    test_data.drop(columns=[col for col in test_data.columns if col not in columns_to_keep], inplace=True)

    X_train = train_data[features]
    y_train = train_data[target]

    X_test = test_data[features]
    y_test = test_data[target]

    return X_train, y_train, X_test, y_test, player_data


def All_NBA_TEAM(stats_players, result_data, year):
    merged_data = pd.merge(stats_players, result_data[['PLAYER_NAME', 'YEAR', 'Result']], on=['PLAYER_NAME', 'YEAR'],
                           how='left')

    merged_data = merged_data.replace({'Result': {'First Team': 3, 'Second Team': 2, 'Third Team': 1}})
    merged_data['Result'] = merged_data['Result'].fillna(0)
    merged_data['Position'] = merged_data['Position'].fillna('NONE')
    merged_data.drop(columns=['TEAM_ID'], inplace=True)
    features = ['AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM',
                'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS',
                'PLUS_MINUS', 'NBA_FANTASY_PTS', 'DD2', 'TD3', 'WNBA_FANTASY_PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT']

    target = 'Result'
    MAIN_FEATURES = corrlation_cal(merged_data, features, year, 0.2)

    filter_data  = filter_top_players_by_year(merged_data, MAIN_FEATURES, 0.4)

    filtered_stats_F = filter_F_position(filter_data)
    filtered_stats_C = filter_C_position(filter_data)
    filtered_stats_G = filter_G_position(filter_data)

    features_F = corrlation_cal(filtered_stats_F, features, year, 0.3)
    features_C = corrlation_cal(filtered_stats_C, features, year, 0.25)
    features_G = corrlation_cal(filtered_stats_G, features, year, 0.1)


    BIG_DATA_F = make_collection(filtered_stats_F, year, target,
                                 features_F)  # X_trainF, y_trainF, X_testF, y_testF, player_dataF
    BIG_DATA_C = make_collection(filtered_stats_C, year, target,
                                 features_C)  # X_trainC, y_trainC, X_testC, y_testC, player_dataC
    BIG_DATA_G = make_collection(filtered_stats_G, year, target,
                                 features_G)  # X_trainG, y_trainG, X_testG, y_testG, player_dataG

    BIG_DATA = [BIG_DATA_F, BIG_DATA_C, BIG_DATA_G]

    models = [
        LogisticRegression(solver='liblinear'),
        KNeighborsClassifier(n_neighbors=8, weights="uniform"),
        RandomForestClassifier(random_state=42, n_estimators=100),
        GradientBoostingClassifier(random_state=42, n_estimators=100),
        MLPClassifier(random_state=42, max_iter=1000),
        AdaBoostClassifier(random_state=42, n_estimators=300),
    ]

    first_team, second_team, third_team = evaluate(models, BIG_DATA)

    filtered_first_team, filtered_second_team, filtered_third_team = remove_duplicate_players(first_team, second_team,
                                                                                              third_team)
    first_ALLteam = list(filtered_first_team.keys())[:5]
    second_ALLteam = list(filtered_second_team.keys())[:5]
    third_ALLteam = list(filtered_third_team.keys())[:5]

    print("FIRST ALL TEAM")
    print(first_ALLteam)
    print("SCOND ALL TEAM")
    print(second_ALLteam)
    print("THIRD ALL TEAM")
    print(third_ALLteam)

    return first_ALLteam, second_ALLteam, third_ALLteam


def All_NBA_ROOKIE_TEAM(stats_players, result_data, year):
    stats_players.sort_values(by=['PLAYER_NAME', 'YEAR'], inplace=True)
    mask = stats_players['YEAR'] == stats_players.groupby('PLAYER_NAME')['YEAR'].transform('min')

    filtered_data = stats_players[mask]
    filtered_data.reset_index(drop=True, inplace=True)

    merged_data = pd.merge(filtered_data, result_data[['PLAYER_NAME', 'YEAR', 'Result']], on=['PLAYER_NAME', 'YEAR'],
                           how='left')
    merged_data = merged_data.replace({'Result': {'First Team': 2, 'Second Team': 1}})
    merged_data['Result'] = merged_data['Result'].fillna(0)
    merged_data.drop(columns=['Position'], inplace=True)
    merged_data.drop(columns=['TEAM_ID'], inplace=True)

    features = ['AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM',
                'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS',
                'PLUS_MINUS', 'NBA_FANTASY_PTS', 'DD2', 'TD3', 'WNBA_FANTASY_PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    target = 'Result'

    featuresOPT = corrlation_cal(merged_data, features, year, 0.35)

    BIG_DATA = make_collection(merged_data, year, target, featuresOPT)  # X_trainF, y_trainF, X_testF, y_testF, player_dataF
    BIG_DATA = [BIG_DATA]
    models = [
        KNeighborsClassifier(n_neighbors=20, weights="uniform"),
        DecisionTreeClassifier(random_state=42),
        AdaBoostClassifier(random_state=42, n_estimators=50),
        SVC(kernel='linear'),
        GaussianNB()
    ]

    _, FIRST, SECOND = evaluate(models, BIG_DATA)

    rookie_first_team = list(FIRST.keys())[:5]
    rookie_second_team = list(SECOND.keys())[:5]

    # print("FIRST ROOKIE TEAM")
    # print(rookie_first_team)
    # print("SECOND ROOKIE TEAM")
    # print(rookie_second_team)

    return rookie_first_team, rookie_second_team

stats_players = pd.read_csv('ALL_position.csv')
result_data = pd.read_csv('result_ALL.csv')
result_data_rookie = pd.read_csv('result_rookie.csv')

numeric_columns = ['AGE', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM',
                   'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS',
                   'PLUS_MINUS', 'NBA_FANTASY_PTS', 'DD2', 'TD3', 'WNBA_FANTASY_PTS']
percentage_columns = ['FG_PCT', 'FG3_PCT', 'FT_PCT']

merged_data = stats_players[stats_players['GP'] >= 65]
scaler = StandardScaler()
stats_players[numeric_columns] = scaler.fit_transform(stats_players[numeric_columns])

scaler = MinMaxScaler()
stats_players[percentage_columns] = scaler.fit_transform(stats_players[percentage_columns])

first_ALL_NBA, second_ALL_NBA, third_ALL_NBA = All_NBA_TEAM(stats_players, result_data, 2023)
rookie_first_NBA_team, rookie_second_NBA_team = All_NBA_ROOKIE_TEAM(stats_players, result_data_rookie, 2023)

generate_all_nba_json(first_ALL_NBA, second_ALL_NBA, third_ALL_NBA, rookie_first_NBA_team, rookie_second_NBA_team, 'nba_teams.json')
