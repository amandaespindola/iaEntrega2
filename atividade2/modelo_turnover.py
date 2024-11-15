import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Carregar os dados
data = pd.read_csv('Dados_RH_Turnover.csv')


import pandas as pd

# Carregar os dados com o delimitador correto
data = pd.read_csv('Dados_RH_Turnover.csv', delimiter=';')

# Verificar as colunas do DataFrame
print(data.columns)



# 2. Pré-processamento dos dados
# Codificar variáveis categóricas (por exemplo, Salário e DeptoAtuacao)
label_encoder = LabelEncoder()
data['Salario'] = label_encoder.fit_transform(data['Salario'])
data['DeptoAtuacao'] = label_encoder.fit_transform(data['DeptoAtuacao'])

# Separar as features e o target
X = data.drop(columns=['SaiuDaEmpresa'])  # Features (todas as colunas exceto o target)
y = data['SaiuDaEmpresa']                # Target (rotatividade: 0 = não saiu, 1 = saiu)

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Treinar e avaliar modelos
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

for model_name, model in models.items():
    # Treinar o modelo
    model.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Avaliar a acurácia e a matriz de confusão
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Modelo: {model_name}")
    print(f"Acurácia: {accuracy:.2f}")
    print("Matriz de Confusão:")
    print(conf_matrix)
    print("\n" + "-"*30 + "\n")
