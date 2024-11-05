import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Carregar os dados
data = pd.read_csv('./creditcard - menor balanceado.csv')

X = data.drop(columns=['Class'])  # Features
y = data['Class']  # Target

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Configuração do SMOTE e undersampling
smote = SMOTE(random_state=42)
undersample = RandomUnderSampler(random_state=42)

# Modelos para ajuste de hiperparâmetros
xgb = XGBClassifier(random_state=42)
svm = SVC(random_state=42)

# Parâmetros para GridSearch e RandomizedSearch
xgb_params = {'n_estimators': [50, 100], 'max_depth': [3, 6, 10], 'learning_rate': [0.01, 0.1, 0.3]}
svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}


# Função para ajustar o modelo e calcular a acurácia
def ajustar_e_calcular_acuracia(modelo, params, X_treinamento, y_treinamento, X_teste, y_teste, tipo_pesquisa):
    if tipo_pesquisa == "grid":
        pesquisa = GridSearchCV(modelo, params, cv=5)
    elif tipo_pesquisa == "random":
        pesquisa = RandomizedSearchCV(modelo, params, cv=5, n_iter=5, random_state=42)

    # Ajustar o modelo
    pesquisa.fit(X_treinamento, y_treinamento)

    # Prever os resultados para calcular a acurácia e outras métricas
    y_pred = pesquisa.best_estimator_.predict(X_teste)

    # Calcular a acurácia
    acuracia = accuracy_score(y_teste, y_pred)

    # Calcular F1-score médio macro
    f1 = f1_score(y_teste, y_pred, average='macro')

    # Calcular matriz de confusão
    matriz_confusao = confusion_matrix(y_teste, y_pred)

    return acuracia, f1, matriz_confusao


# Dicionário para armazenar as acurácias, F1 e matrizes de confusão
resultados = {}

# 1. RFE (Recursive Feature Elimination) para selecionar 10 features
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
X_rfe = rfe.fit_transform(X_train, y_train)

# Aplicar SMOTE e undersampling nos dados RFE
X_rfe_resampled, y_rfe_resampled = smote.fit_resample(X_rfe, y_train)
X_rfe_resampled, y_rfe_resampled = undersample.fit_resample(X_rfe_resampled, y_rfe_resampled)

# Calcular a acurácia com RFE
resultados["XGBoost com RFE (Grid)"] = ajustar_e_calcular_acuracia(xgb, xgb_params, X_rfe_resampled, y_rfe_resampled,
                                                                   X_rfe, y_train, "grid")
resultados["XGBoost com RFE (Random)"] = ajustar_e_calcular_acuracia(xgb, xgb_params, X_rfe_resampled, y_rfe_resampled,
                                                                     X_rfe, y_train, "random")
resultados["SVM com RFE (Grid)"] = ajustar_e_calcular_acuracia(svm, svm_params, X_rfe_resampled, y_rfe_resampled, X_rfe,
                                                               y_train, "grid")
resultados["SVM com RFE (Random)"] = ajustar_e_calcular_acuracia(svm, svm_params, X_rfe_resampled, y_rfe_resampled,
                                                                 X_rfe, y_train, "random")

# 2. Random Forest para seleção de features
rf_selector = RandomForestClassifier(random_state=42)
rf_selector.fit(X_train, y_train)

# Obter as importâncias das features e selecionar as 10 mais importantes
importancias = rf_selector.feature_importances_
indices_importantes = importancias.argsort()[-10:][::-1]
X_rf = X_train.iloc[:, indices_importantes]

# Aplicar SMOTE e undersampling nos dados da Random Forest
X_rf_resampled, y_rf_resampled = smote.fit_resample(X_rf, y_train)
X_rf_resampled, y_rf_resampled = undersample.fit_resample(X_rf_resampled, y_rf_resampled)

# Calcular a acurácia com Random Forest
resultados["XGBoost com Random Forest (Grid)"] = ajustar_e_calcular_acuracia(xgb, xgb_params, X_rf_resampled,
                                                                             y_rf_resampled, X_rf, y_train, "grid")
resultados["XGBoost com Random Forest (Random)"] = ajustar_e_calcular_acuracia(xgb, xgb_params, X_rf_resampled,
                                                                               y_rf_resampled, X_rf, y_train, "random")
resultados["SVM com Random Forest (Grid)"] = ajustar_e_calcular_acuracia(svm, svm_params, X_rf_resampled,
                                                                         y_rf_resampled, X_rf, y_train, "grid")
resultados["SVM com Random Forest (Random)"] = ajustar_e_calcular_acuracia(svm, svm_params, X_rf_resampled,
                                                                           y_rf_resampled, X_rf, y_train, "random")

# 3. SelectKBest para selecionar 10 melhores features
k_best = SelectKBest(k=10)
X_kbest = k_best.fit_transform(X_train, y_train)

# Aplicar SMOTE e undersampling nos dados SelectKBest
X_kbest_resampled, y_kbest_resampled = smote.fit_resample(X_kbest, y_train)
X_kbest_resampled, y_kbest_resampled = undersample.fit_resample(X_kbest_resampled, y_kbest_resampled)

# Calcular a acurácia com SelectKBest
resultados["XGBoost com SelectKBest (Grid)"] = ajustar_e_calcular_acuracia(xgb, xgb_params, X_kbest_resampled,
                                                                           y_kbest_resampled, X_kbest, y_train, "grid")
resultados["XGBoost com SelectKBest (Random)"] = ajustar_e_calcular_acuracia(xgb, xgb_params, X_kbest_resampled,
                                                                             y_kbest_resampled, X_kbest, y_train,
                                                                             "random")
resultados["SVM com SelectKBest (Grid)"] = ajustar_e_calcular_acuracia(svm, svm_params, X_kbest_resampled,
                                                                       y_kbest_resampled, X_kbest, y_train, "grid")
resultados["SVM com SelectKBest (Random)"] = ajustar_e_calcular_acuracia(svm, svm_params, X_kbest_resampled,
                                                                         y_kbest_resampled, X_kbest, y_train, "random")

# Imprimir os resultados
for descricao, (acuracia, f1, matriz) in resultados.items():
    print(f"{descricao}:")
    print(f"  Acurácia: {acuracia}")
    print(f"  F1-score (macro avg): {f1}")
    print("  Matriz de Confusão:")
    print(matriz)
    print()

# Salvando os resultados de acurácia em CSV
pd.DataFrame(resultados.items(), columns=["Descrição", "Resultados"]).to_csv('resultados_hiperparametros.csv',
                                                                             index=False)
