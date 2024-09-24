import random
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Função para extrair características de um nome
def extrair_caracteristicas(nome):
    return {
        'ultima_letra': nome[-1].lower(),
        'primeira_letra': nome[0].lower(),
        'tamanho_nome': len(nome),
        'has_vogal': any(letra in 'aeiou' for letra in nome.lower())
    }

# Listas de nomes masculinos e femininos
nomes_masculinos = ['João', 'Pedro', 'Lucas', 'Mateus', 'Bruno', 'Gabriel', 'Thiago', 'Rafael']
nomes_femininos = ['Maria', 'Ana', 'Cláudia', 'Juliana', 'Fernanda', 'Patrícia', 'Carla', 'Mariana']

# Criando um conjunto de dados com rótulos
dados = [(nome, 'masculino') for nome in nomes_masculinos] + [(nome, 'feminino') for nome in nomes_femininos]

# Embaralhar os dados
random.shuffle(dados)

# Separando características e rótulos
caracteristicas = [extrair_caracteristicas(nome) for nome, genero in dados]
rotulos = [genero for nome, genero in dados]

# Vetorizando características
vetorizador = DictVectorizer()
X = vetorizador.fit_transform(caracteristicas)

# Dividindo os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, rotulos, test_size=0.3, random_state=42)

# Treinando o classificador
classificador = MultinomialNB()
classificador.fit(X_train, y_train)

# Fazendo previsões
previsoes = classificador.predict(X_test)

# Exibindo a precisão do modelo
print(f"Precisão do modelo: {metrics.accuracy_score(y_test, previsoes) * 100:.2f}%")

# Testando com novos nomes
nomes_teste = ['Felipe', 'Beatriz', 'Carlos', 'Aline']
caracteristicas_teste = [extrair_caracteristicas(nome) for nome in nomes_teste]
X_teste = vetorizador.transform(caracteristicas_teste)

previsoes_teste = classificador.predict(X_teste)

# Exibindo os resultados do teste
for nome, previsao in zip(nomes_teste, previsoes_teste):
    print(f"Nome: {nome} | Previsão: {previsao}")
