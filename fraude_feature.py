import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data = pd.read_csv('./creditcard - menor balanceado.csv')

X = data.drop(columns=['Class'])  # Features
y = data['Class']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)  # Dados de treino e teste

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

        # Usando SelectKBest
        seletor_kbest = SelectKBest(score_func=f_classif, k=k)
        X_train_kbest = seletor_kbest.fit_transform(X_train, y_train)
        X_test_kbest = seletor_kbest.transform(X_test)

        # Usando RFE e Decision Tree
        modelo_rfe = DecisionTreeClassifier(random_state=1)
        rfe = RFE(estimator=modelo_rfe, n_features_to_select=n_features)
        X_train_rfe = rfe.fit_transform(X_train, y_train)
        X_test_rfe = rfe.transform(X_test)

        # Usando Random Forest e Feature Importance
        modelo_rf = RandomForestClassifier(random_state=1)
        modelo_rf.fit(X_train, y_train)
        importances = modelo_rf.feature_importances_

        # Criando dataframe para mostrar importances
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        })

        # Selecionando as melhores features com base nas importâncias
        melhores_features = feature_importance_df.nlargest(n_features, 'Importance')['Feature']
        X_train_rf = X_train[melhores_features]
        X_test_rf = X_test[melhores_features]

        # Parte de treinamento e avaliação de modelos
        # 1. SelectKBest
        clf_kbest = RandomForestClassifier(random_state=1)
        clf_kbest.fit(X_train_kbest, y_train)
        y_pred_kbest = clf_kbest.predict(X_test_kbest)

        # 2. RFE
        clf_rfe = RandomForestClassifier(random_state=1)
        clf_rfe.fit(X_train_rfe, y_train)
        y_pred_rfe = clf_rfe.predict(X_test_rfe)

        # 3. Random Forest
        clf_rf = RandomForestClassifier(random_state=1)
        clf_rf.fit(X_train_rf, y_train)
        y_pred_rf = clf_rf.predict(X_test_rf)

        # Acurácias de cada modelo
        acuracia_kbest = accuracy_score(y_test, y_pred_kbest)
        acuracia_rfe = accuracy_score(y_test, y_pred_rfe)
        acuracia_rf = accuracy_score(y_test, y_pred_rf)

        # Armazenando os resultados
        resultados.append({
            'k': k,
            'n_features': n_features,
            'acuracia_kbest': acuracia_kbest,
            'acuracia_rfe': acuracia_rfe,
            'acuracia_rf': acuracia_rf,
            'melhor_modelo': max(
                ('SelectKBest', acuracia_kbest),
                ('RFE', acuracia_rfe),
                ('Random Forest', acuracia_rf),
                key=lambda x: x[1]
            )
        })

# Imprimindo os resultados de todos os testes
for i, resultado in enumerate(resultados):
    print(f"Teste {i + 1}:")
    print(f"Acurácia com SelectKBest: {resultado['acuracia_kbest']:.4f}")
    print(f"Acurácia com RFE: {resultado['acuracia_rfe']:.4f}")
    print(f"Acurácia com Random Forest: {resultado['acuracia_rf']:.4f}")
    print(f"Melhor modelo: {resultado['melhor_modelo'][0]} com acurácia: {resultado['melhor_modelo'][1]:.4f}")
    print()

# Salvando os resultados em um CSV
arquivo = 'resultados_teste_features.csv'
first_row = not os.path.exists(arquivo)  # Vendo se já existe

with open(arquivo, 'a') as f:
    if first_row:
        # Escrever o cabeçalho se for a primeira vez
        f.write('Teste,k,n_features,Acurácia SelectKBest,Acurácia RFE,Acurácia Random Forest,Melhor Modelo,Melhor Acurácia\n')
    for i, resultado in enumerate(resultados):
        f.write(f'{i + 1},{resultado["k"]},{resultado["n_features"]},{resultado["acuracia_kbest"]},{resultado["acuracia_rfe"]},{resultado["acuracia_rf"]},{resultado["melhor_modelo"][0]},{resultado["melhor_modelo"][1]}\n')

print(f'Resultados salvos em {arquivo}')