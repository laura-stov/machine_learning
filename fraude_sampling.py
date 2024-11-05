# OBSERVAÇÕES:
# 1. pip install imbalanced-learn

import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd

data = pd.read_csv('./creditcard - menor balanceado.csv')

X = data.drop(columns=['Class'])  # features
y = data['Class']  # target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)  # dados de treino e teste

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

        # Usando Random Undersampling
        undersampler = RandomUnderSampler(random_state=1)
        X_train_undersampled, y_train_undersampled = undersampler.fit_resample(X_train, y_train)

        # Usando SMOTE
        smote = SMOTE(random_state=1)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Usando SelectKBest com dados subamostrados
        seletor_kbest = SelectKBest(score_func=f_classif, k=k)
        X_train_kbest = seletor_kbest.fit_transform(X_train_undersampled, y_train_undersampled)
        X_test_kbest = seletor_kbest.transform(X_test)

        # Usando RFE com dados subamostrados
        modelo_rfe = DecisionTreeClassifier(random_state=1)
        rfe = RFE(estimator=modelo_rfe, n_features_to_select=n_features)
        X_train_rfe = rfe.fit_transform(X_train_undersampled, y_train_undersampled)
        X_test_rfe = rfe.transform(X_test)

        # Usando Random Forest e Feature Importance
        modelo_rf = RandomForestClassifier(random_state=1)
        modelo_rf.fit(X_train_undersampled, y_train_undersampled)
        importances = modelo_rf.feature_importances_

        # Criando dataframe para mostrar importances
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        })

        # Selecionando as melhores features com base nas importâncias
        melhores_features = feature_importance_df.nlargest(n_features, 'Importance')['Feature']
        X_train_rf = X_train_undersampled[melhores_features]
        X_test_rf = X_test[melhores_features]

        # Parte de treinamento e avaliação de modelos
        # 1. SelectKBest com SMOTE
        clf_kbest_smote = RandomForestClassifier(random_state=1)
        clf_kbest_smote.fit(X_train_kbest, y_train_undersampled)
        y_pred_kbest_smote = clf_kbest_smote.predict(X_test_kbest)

        # 2. RFE com SMOTE
        clf_rfe_smote = RandomForestClassifier(random_state=1)
        clf_rfe_smote.fit(X_train_rfe, y_train_undersampled)
        y_pred_rfe_smote = clf_rfe_smote.predict(X_test_rfe)

        # 3. Random Forest com SMOTE
        clf_rf_smote = RandomForestClassifier(random_state=1)
        clf_rf_smote.fit(X_train_rf, y_train_undersampled)
        y_pred_rf_smote = clf_rf_smote.predict(X_test_rf)

        # Acurácias de cada modelo com SMOTE
        acuracia_kbest_smote = accuracy_score(y_test, y_pred_kbest_smote)
        acuracia_rfe_smote = accuracy_score(y_test, y_pred_rfe_smote)
        acuracia_rf_smote = accuracy_score(y_test, y_pred_rf_smote)

        # 4. SelectKBest com Undersampling
        clf_kbest_undersampled = RandomForestClassifier(random_state=1)
        clf_kbest_undersampled.fit(X_train_kbest, y_train_undersampled)
        y_pred_kbest_undersampled = clf_kbest_undersampled.predict(X_test_kbest)

        # 5. RFE com Undersampling
        clf_rfe_undersampled = RandomForestClassifier(random_state=1)
        clf_rfe_undersampled.fit(X_train_rfe, y_train_undersampled)
        y_pred_rfe_undersampled = clf_rfe_undersampled.predict(X_test_rfe)

        # 6. Random Forest com Undersampling
        clf_rf_undersampled = RandomForestClassifier(random_state=1)
        clf_rf_undersampled.fit(X_train_rf, y_train_undersampled)
        y_pred_rf_undersampled = clf_rf_undersampled.predict(X_test_rf)

        # Acurácias de cada modelo com Undersampling
        acuracia_kbest_undersampled = accuracy_score(y_test, y_pred_kbest_undersampled)
        acuracia_rfe_undersampled = accuracy_score(y_test, y_pred_rfe_undersampled)
        acuracia_rf_undersampled = accuracy_score(y_test, y_pred_rf_undersampled)

        # Armazenando os resultados
        resultados.append({
            'k': k,
            'n_features': n_features,
            'acuracia_kbest_smote': acuracia_kbest_smote,
            'acuracia_rfe_smote': acuracia_rfe_smote,
            'acuracia_rf_smote': acuracia_rf_smote,
            'acuracia_kbest_undersampled': acuracia_kbest_undersampled,
            'acuracia_rfe_undersampled': acuracia_rfe_undersampled,
            'acuracia_rf_undersampled': acuracia_rf_undersampled,
        })

# Imprimindo os resultados de todos os testes
for i, resultado in enumerate(resultados):
    print(f"Teste {i + 1}:")
    print(f"Acurácia com SelectKBest (SMOTE): {resultado['acuracia_kbest_smote']}")
    print(f"Acurácia com RFE (SMOTE): {resultado['acuracia_rfe_smote']}")
    print(f"Acurácia com Random Forest (SMOTE): {resultado['acuracia_rf_smote']}")
    print(f"Acurácia com SelectKBest (Undersampling): {resultado['acuracia_kbest_undersampled']}")
    print(f"Acurácia com RFE (Undersampling): {resultado['acuracia_rfe_undersampled']}")
    print(f"Acurácia com Random Forest (Undersampling): {resultado['acuracia_rf_undersampled']}")
    print()

# Salvando os resultados em um CSV
arquivo = 'resultados_teste_features_com_balanceamento.csv'
first_row = not os.path.exists(arquivo)  # vendo se já existe

with open(arquivo, 'a') as f:
    if first_row:
        # escrever o cabeçalho se for a primeira vez
        f.write('Teste,k,n_features,Acurácia SelectKBest SMOTE,Acurácia RFE SMOTE,Acurácia Random Forest SMOTE,Acurácia SelectKBest Undersampling,Acurácia RFE Undersampling,Acurácia Random Forest Undersampling\n')
    for i, resultado in enumerate(resultados):
        f.write(f'{i + 1},{resultado["k"]},{resultado["n_features"]},{resultado["acuracia_kbest_smote"]},{resultado["acuracia_rfe_smote"]},{resultado["acuracia_rf_smote"]},{resultado["acuracia_kbest_undersampled"]},{resultado["acuracia_rfe_undersampled"]},{resultado["acuracia_rf_undersampled"]}\n')

print(f'Resultados salvos em {arquivo}')
