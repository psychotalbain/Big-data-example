from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Dados de carros e suas respectivas marcas
car_data = [
    ["Chevrolet", "Bolt"], ["Chevrolet", "Cruze"], ["Chevrolet", "Equinox"],
    ["Chevrolet", "Montana"], ["Chevrolet", "Onix"], ["Chevrolet", "S10"],
    ["Chevrolet", "Silverado"], ["Chevrolet", "Spin"], 
    ["Fiat", "Uno"], ["Fiat", "Palio"], ["Fiat", "Siena"], 
    ["Fiat", "Argo"], ["Fiat", "Cronos"], ["Fiat", "Mobi"], ["Fiat", "Ducato"],
    ["Volkswagen", "AMAROK"], ["Volkswagen", "ID.4"], ["Volkswagen", "ID.BUZZ"],
    ["Volkswagen", "JETTA"], ["Volkswagen", "NIVUS"], ["Volkswagen", "POLO"], 
    ["Volkswagen", "SAVEIRO"], ["Volkswagen", "T-CROSS"],
    ["Renault", "KARDIAN"], ["Renault", "KWID"], ["Renault", "DUSTER"], 
    ["Renault", "STEPWAY"], ["Renault", "LOGAN"], ["Renault", "OROCH"], 
    ["Renault", "MEGANE E-TECH"], ["Renault", "KWID E-TECH"]
]

# Separando as marcas (target) e os modelos (features)
brands = [car[0] for car in car_data]  # Lista de marcas
models = [car[1] for car in car_data]  # Lista de modelos

# Codificando os modelos de carros em números
label_encoder = LabelEncoder()
models_encoded = label_encoder.fit_transform(models)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(models_encoded.reshape(-1, 1), brands, test_size=0.3, random_state=42)

# Criando e treinando o classificador RandomForest
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Prevendo os resultados no conjunto de teste
y_pred = classifier.predict(X_test)

# Medindo a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

# Mostrando as previsões
print("\nPrevisões para o conjunto de teste:")
for i in range(len(X_test)):
    modelo_original = label_encoder.inverse_transform([X_test[i][0]])[0]
    print(f"Modelo de carro: {modelo_original} -> Previsto: {y_pred[i]} | Verdadeiro: {y_test[i]}")

# Adicionando etapa de teste: Entradas de novos modelos para previsão
print("\n--- Testando com novos modelos de carros ---")
novos_modelos = ["Onix", "Argo", "DUSTER", "JETTA", "KWID"]

# Convertendo os novos modelos para a codificação usada
novos_modelos_encoded = label_encoder.transform(novos_modelos)

# Realizando previsões nos novos modelos
novas_previsoes = classifier.predict(novos_modelos_encoded.reshape(-1, 1))

# Mostrando as previsões para os novos modelos
print("\nPrevisões para novos modelos de carros:")
for i in range(len(novos_modelos)):
    print(f"Modelo de carro: {novos_modelos[i]} -> Previsto: {novas_previsoes[i]}")
