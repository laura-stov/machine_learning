import os
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
import xgboost as xgb

# Carregando os dados
data = pd.read_csv('./creditcard - menor balanceado.csv')

X = data.drop(columns=['Class'])  # features
y = data['Class']  # target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)  # dados de treino e teste

# Usando SMOTE
smote = SMOTE(random_state=1)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Dados antes da modificação
clf = DecisionTreeClassifier(random_state=1)
clf.fit(X_train, y_train)
y_prev = clf.predict(X_test)

print('Acurácia antes de usar técnicas: ', accuracy_score(y_test, y_prev))
print('Matriz de confusão antes das técnicas:\n', confusion_matrix(y_test, y_prev))
print()

# -------------------------------------
# Testando diferentes valores de k em SelectKBest e n_features em RFE
resultados = []

# Teste de valores para SelectKBest e RFE
for k in [5, 10, 15, 20, 25]:  # Valores de k para SelectKBest
    for n_features in [5, 10, 15, 20, 25]:  # Valores para n_features em RFE

        # Usando SelectKBest com dados balanceados
        seletor_kbest = SelectKBest(score_func=f_classif, k=k)
        X_train_kbest = seletor_kbest.fit_transform(X_train_smote, y_train_smote)
        X_test_kbest = seletor_kbest.transform(X_test)

        # Usando RFE com dados balanceados
        modelo_rfe = DecisionTreeClassifier(random_state=1)
        rfe = RFE(estimator=modelo_rfe, n_features_to_select=n_features)
        X_train_rfe = rfe.fit_transform(X_train_smote, y_train_smote)

        # Transformando o conjunto de teste para ter as mesmas features que o conjunto de treino
        X_test_rfe = rfe.transform(X_test)

        # 1. SelectKBest com SMOTE
        clf_kbest_smote = RandomForestClassifier(random_state=1)
        clf_kbest_smote.fit(X_train_kbest, y_train_smote)
        y_pred_kbest_smote = clf_kbest_smote.predict(X_test_kbest)

        # 2. RFE com SMOTE
        clf_rfe_smote = RandomForestClassifier(random_state=1)
        clf_rfe_smote.fit(X_train_rfe, y_train_smote)
        y_pred_rfe_smote = clf_rfe_smote.predict(X_test_rfe)

        # Acurácias de cada modelo com SMOTE
        acuracia_kbest_smote = accuracy_score(y_test, y_pred_kbest_smote)
        acuracia_rfe_smote = accuracy_score(y_test, y_pred_rfe_smote)

        # Armazenando os resultados
        resultados.append({
            'k': k,
            'n_features': n_features,
            'acuracia_kbest_smote': acuracia_kbest_smote,
            'acuracia_rfe_smote': acuracia_rfe_smote,
        })

# Imprimindo os resultados de todos os testes
for i, resultado in enumerate(resultados):
    print(f"Teste {i + 1}:")
    print(f"Acurácia com SelectKBest (SMOTE): {resultado['acuracia_kbest_smote']:.4f}")
    print(f"Acurácia com RFE (SMOTE): {resultado['acuracia_rfe_smote']:.4f}")
    print()

# Salvando os resultados em um CSV
arquivo_resultados = 'resultados_teste_features_com_balanceamento.csv'
first_row = not os.path.exists(arquivo_resultados)  # vendo se já existe

with open(arquivo_resultados, 'a') as f:
    if first_row:
        # escrever o cabeçalho se for a primeira vez
        f.write('Teste,k,n_features,Acurácia SelectKBest SMOTE,Acurácia RFE SMOTE\n')
    for i, resultado in enumerate(resultados):
        f.write(f'{i + 1},{resultado["k"]},{resultado["n_features"]},{resultado["acuracia_kbest_smote"]},{resultado["acuracia_rfe_smote"]}\n')

print(f'Resultados salvos em {arquivo_resultados}')

# Ajustes finos nos hiperparâmetros dos modelos usando GridSearchCV
# Exemplo para Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=1), param_grid_rf, cv=3)
grid_search_rf.fit(X_train_smote, y_train_smote)
print(f"Melhor acurácia com GridSearchCV para Random Forest: {grid_search_rf.best_score_}")

# Ajustes finos nos hiperparâmetros dos modelos usando RandomizedSearchCV
# Exemplo para XGBoost
param_distributions_xgb = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

random_search_xgb = RandomizedSearchCV(xgb.XGBClassifier(eval_metric='logloss'), param_distributions_xgb, n_iter=10, cv=3, random_state=1)
random_search_xgb.fit(X_train_smote, y_train_smote)
print(f"Melhor acurácia com RandomizedSearchCV para XGBoost: {random_search_xgb.best_score_}")

# Salvando os resultados das melhores acurácias em um novo arquivo CSV
arquivo_hiperparametros = 'resultados_hiperparametros.csv'
first_row = not os.path.exists(arquivo_hiperparametros)  # vendo se já existe

with open(arquivo_hiperparametros, 'a') as f:
    if first_row:
        # escrever o cabeçalho se for a primeira vez
        f.write('Modelo,Melhor Acurácia\n')
    f.write(f'Random Forest,{grid_search_rf.best_score_}\n')
    f.write(f'XGBoost,{random_search_xgb.best_score_}\n')

print(f'Resultados de hiperparâmetros salvos em {arquivo_hiperparametros}')

# Criação dos modelos para o Voting Classifier
clf_dt = DecisionTreeClassifier(random_state=1)
clf_rf = RandomForestClassifier(random_state=1)
clf_svc = SVC(probability=True, random_state=1)  # SVC com probabilidade para Voting
clf_knn = KNeighborsClassifier()

# Ensemble com Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('dt', clf_dt),
    ('rf', clf_rf),
    ('svc', clf_svc),
    ('knn', clf_knn)
], voting='soft')  # 'soft' para usar as probabilidades

# Treinando o Voting Classifier
voting_clf.fit(X_train_smote, y_train_smote)

# Fazendo previsões
y_pred_voting = voting_clf.predict(X_test)

# Acurácia do Voting Classifier
acuracia_voting = accuracy_score(y_test, y_pred_voting)

print(f"Acurácia do Voting Classifier: {acuracia_voting:.4f}")
print('Matriz de confusão:\n', confusion_matrix(y_test, y_pred_voting))
print()

# Salvando os resultados em um novo arquivo CSV para o ensemble
arquivo_ensemble = 'resultados_ensemble.csv'
first_row = not os.path.exists(arquivo_ensemble)  # vendo se já existe

with open(arquivo_ensemble, 'a') as f:
    if first_row:
        # escrever o cabeçalho se for a primeira vez
        f.write('Modelo,Acurácia\n')
    f.write(f'Voting Classifier,{acuracia_voting:.4f}\n')

print(f'Resultados do ensemble salvos em {arquivo_ensemble}')
