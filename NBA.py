import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


def plot_correlation_pearson_bar_sorted(data, features, year, value):
    data = data[data['YEAR'] != year].copy()
    correlation_values = {}
    for feature in features:
        correlation, _ = spearmanr(data[feature], data['Result'])
        correlation_values[feature] = correlation

    sorted_correlation = sorted(correlation_values.items(), key=lambda x: x[1], reverse=True)
    sorted_features = [x[0] for x in sorted_correlation]
    sorted_correlation_values = [x[1] for x in sorted_correlation]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_features, y=sorted_correlation_values, color='skyblue')
    plt.title('Correlation between Features and Result (Sorted)')
    plt.xlabel('Feature')
    plt.ylabel('Correlation with Result')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    significant_features = [feature for feature, correlation in sorted_correlation if correlation >= value]
    return significant_features


def plot_correlation_bar_sorted(data, features, year, value):
    data = data[data['YEAR'] != year].copy()
    correlation_matrix = data[features + ['Result']].corr()
    sorted_correlation = correlation_matrix['Result'].drop('Result').sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sorted_correlation.plot(kind='bar', color='skyblue')
    plt.title('Correlation between Features and Result (Sorted)')
    plt.xlabel('Feature')
    plt.ylabel('Correlation with Result')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    significant_features = sorted_correlation[sorted_correlation >= value].index.tolist()
    return significant_features


def corr_map(df, f):

    data_for_corr = df[f]
    corr = data_for_corr.corr()
    plt.figure(figsize=(30, 15))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Map COR')
    plt.show()

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
        print('model:', model)
        for data in BIG_DATA:
            X_train, y_train, X_test, y_test, player_data = data
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            player_count_100 = {} #first team
            player_count_60 = {} #second team
            player_count_20 = {} #third team

            for player_name, pred_label in zip(player_data['PLAYER_NAME'], y_pred):
                if pred_label == 3:
                    player_count_100[player_name] = player_count_100.get(player_name, 0) + 1
                if pred_label == 2:
                    player_count_60[player_name] = player_count_60.get(player_name, 0) + 1
                if pred_label == 1:
                    player_count_20[player_name] = player_count_20.get(player_name, 0) + 1
            print("player_count_3: ", player_count_100)
            print("player_count_2: ", player_count_60)
            print("player_count_1: ", player_count_20)
            print(" ")

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

    return X_train, y_train, X_test,  y_test, player_data

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


    features_G = ['PTS', 'AST', 'TOV', 'L', 'BLK', 'STL'] # idealnie
    features_C = ['REB', 'PTS', 'GP' , 'PLUS_MINUS', 'STL', 'REB',]
    features_F = ['PTS', 'PLUS_MINUS', 'AST', 'GP', 'STL', 'REB', 'BLK', 'L' ]# idealnie

    target = 'Result'

    filtered_stats_F = filter_F_position(merged_data)
    filtered_stats_C = filter_C_position(merged_data)
    filtered_stats_G = filter_G_position(merged_data)

    features_F = plot_correlation_bar_sorted(filtered_stats_F, features, year, 0.3)
    features_C = plot_correlation_bar_sorted(filtered_stats_C, features, year, 0.3)
    features_G = plot_correlation_bar_sorted(filtered_stats_G, features, year, 0.3)

    #corr_map(filtered_stats_F, features_F)
    BIG_DATA_F = make_collection(filtered_stats_F, year, target, features_F)#X_trainF, y_trainF, X_testF, y_testF, player_dataF
    BIG_DATA_C = make_collection(filtered_stats_C, year, target, features_C)#X_trainC, y_trainC, X_testC, y_testC, player_dataC
    BIG_DATA_G = make_collection(filtered_stats_G, year, target, features_G)#X_trainG, y_trainG, X_testG, y_testG, player_dataG

    BIG_DATA = [BIG_DATA_F, BIG_DATA_C , BIG_DATA_G]
    #BIG_DATA = [BIG_DATA_F]
    models = [
        LogisticRegression(random_state=42), #4,0,1  #ten
        KNeighborsClassifier(n_neighbors=8, weights="uniform"), #3,0,0#c
        #DecisionTreeClassifier(random_state=42),#1,0,0
        RandomForestClassifier(random_state=42, n_estimators=100),#3,0,0#c
        #GradientBoostingClassifier(random_state=42, n_estimators=100), #2,0,0#ten
        AdaBoostClassifier(random_state=42), #ten #1,1,0
        SVC(kernel='linear'), #3,0,0
        #GaussianNB(), #2,0,0
        MLPClassifier(random_state=42, max_iter=1000) #3,1,1
    ]

    total_player_count_100, total_player_count_60, total_player_count_20 = evaluate(models, BIG_DATA)

    print("FIRST TEAM")
    print(list(total_player_count_100.items())[:5])
    print("SECOND TEAM")
    print(list(total_player_count_60.items())[:5])
    print("THIRD TEAM")
    print(list(total_player_count_20.items())[:5])

    data_2023 = merged_data[merged_data['YEAR'] == year]

    # Wyświetlenie graczy z różnymi wartościami wynikowymi
    first_team = set(data_2023[data_2023['Result'] == 3]['PLAYER_NAME'])
    second_team = set(data_2023[data_2023['Result'] == 2]['PLAYER_NAME'])
    third_team = set(data_2023[data_2023['Result'] == 1]['PLAYER_NAME'])

    top_5_first_team = set(list(total_player_count_100.keys())[:5])
    top_5_second_team = set(list(total_player_count_60.keys())[:5])
    top_5_third_team = set(list(total_player_count_20.keys())[:5])

    common_in_first = first_team.intersection(top_5_first_team)
    common_in_second = second_team.intersection(top_5_second_team)
    common_in_third = third_team.intersection(top_5_third_team)

    print("Liczba dopasowań w pierwszej drużynie:", len(common_in_first))
    print("Liczba dopasowań w drugiej drużynie:", len(common_in_second))
    print("Liczba dopasowań w trzeciej drużynie:", len(common_in_third))

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

    featuresOPT = plot_correlation_bar_sorted(merged_data, features, year, 0.35)
    #featuresOPT = plot_correlation_pearson_bar_sorted(merged_data, features, year, 0.3)
    #print(featuresOPT)
    corr_map( merged_data, featuresOPT)
    BIG_DATA = make_collection(merged_data, year, target, featuresOPT)#X_trainF, y_trainF, X_testF, y_testF, player_dataF
    BIG_DATA = [BIG_DATA]
    models = [
        KNeighborsClassifier(n_neighbors=20, weights="uniform"),
        DecisionTreeClassifier(random_state=42),
        AdaBoostClassifier(random_state=42, n_estimators=50),
        SVC(kernel='linear'),
        GaussianNB()
    ]

    total_player_count_100, total_player_count_60, total_player_count_20 = evaluate(models, BIG_DATA)

    print("FIRST TEAM")
    print(list(total_player_count_100.items())[:5])
    print("SECOND TEAM")
    print(list(total_player_count_60.items())[:5])
    print("THIRD TEAM")
    print(list(total_player_count_20.items())[:5])

    # Wybór danych dla roku 2023
    data_2023 = merged_data[merged_data['YEAR'] == 2023]

    # Wyświetlenie graczy z różnymi wartościami wynikowymi
    first_team = set(data_2023[data_2023['Result'] == 3]['PLAYER_NAME'])
    second_team = set(data_2023[data_2023['Result'] == 2]['PLAYER_NAME'])
    third_team = set(data_2023[data_2023['Result'] == 1]['PLAYER_NAME'])

    # Pobranie pierwszych pięciu kluczy (imion graczy) z każdego słownika
    top_5_first_team = set(list(total_player_count_100.keys())[:5])
    top_5_second_team = set(list(total_player_count_60.keys())[:5])
    top_5_third_team = set(list(total_player_count_20.keys())[:5])

    # Liczenie dopasowań
    common_in_first = first_team.intersection(top_5_first_team)
    common_in_second = second_team.intersection(top_5_second_team)
    common_in_third = third_team.intersection(top_5_third_team)

    # Wyświetlenie liczby dopasowań
    print("Liczba dopasowań w pierwszej drużynie:", len(common_in_first))
    print("Liczba dopasowań w drugiej drużynie:", len(common_in_second))
    print("Liczba dopasowań w trzeciej drużynie:", len(common_in_third))


stats_players = pd.read_csv('ALL_position.csv')
result_data = pd.read_csv('result_ALL_NBA_Team.csv')
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

All_NBA_TEAM(stats_players, result_data, 2023)
#All_NBA_ROOKIE_TEAM(stats_players, result_data_rookie, 2023)