import pandas as pd
from random import choice
from mcts import MCTS, Node
import hashlib
import sys

def limpar_dados(data):
    print("limpando dados...")
    # LIMPAR NOMES DAS COLUNAS, REMOVER ACENTOS E CARACTERES ESPECIAIS E REMOVER ESPAÇOS EM BRANCO
    data.columns = data.columns.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

    #Limpar os dados das linhas REMOVER ACENTOS E CARACTERES ESPECIAIS E REMOVER ESPAÇOS EM BRANCO
    data = data.apply(lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8') if x.dtype == "object" else x)
    #remover espaços em branco
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    return data

def definir_condicoes():
        conditions = []

        colunas_ignoradas = ['NUMERO_BOLETIM', 'DATA HORA_BOLETIM', 'DATA_INCLUSAO', 'TIPO_ACIDENTE', 'COD_TEMPO', 'COD_PAVIMENTO', 'COD_REGIONAL', 'ORIGEM_BOLETIM', 'COORDENADA_X', 'COORDENADA_Y', 'HORA_INFORMADA', 'VALOR_UPS', 'DESCRICAO_UPS', 'DATA_ALTERACAO_SMSA', 'VALOR_UPS_ANTIGA', 'DESCRICAO_UPS_ANTIGA', 'DESCRIAAO_UPS', 'DESCRIAAO_UPS_ANTIGA']


        for column in data.columns:
            if column in colunas_ignoradas:
                continue
            for value in data[column].unique():
                conditions.append((column, value))
        return conditions
class AccidentNode(Node):
    def __init__(self, data, conditions=None):
        self.data = data
        self.conditions = conditions if conditions else []
        self.children = None
        self._hash = None  # Cache do hash
        self.conditionsDefinidas = definir_condicoes()

    def find_children(self):
        if self.is_terminal():
            return set()
        if self.children is None:
            self.children = set()
            possible_conditions = self.conditionsDefinidas
            for condition in possible_conditions:
                new_conditions = self.conditions + [condition]
                new_data = self.apply_conditions(self.data, new_conditions)
                self.children.add(AccidentNode(new_data, new_conditions))
                print(f"Novo nó criado com condições: {new_conditions}, Dados restantes: {new_data.shape}")
        return self.children

    def find_random_child(self):
        children = list(self.find_children())
        if not children:
            return None
        return choice(children)

    def is_terminal(self):
        # Um nó é terminal se não houver dados após a aplicação das condições
        # print("is_terminal->", self.data.empty, "\n dados: \n", self.data)
        return self.data.empty

    def reward(self):
        # print("reward->", self.data.head())
        if not self.is_terminal():
            raise RuntimeError("reward called on nonterminal node")
        return self.calculate_reward(self.data)

    def apply_conditions(self, data, conditions):
        filtered_data = data.copy(deep=False)  # Use cópia rasa para eficiência
        for column, value in conditions:
            filtered_data = filtered_data[filtered_data[column] == value]
            print(f"Condição aplicada: {column} == {value}, Dados restantes: {filtered_data.shape}")
        return filtered_data

    def calculate_reward(self, data):
        
        print("\nCalculando recompensa... -> ", -data['NUM_ACIDENTES'].sum())
        print(data)
        return -data['NUM_ACIDENTES'].sum()  # Queremos minimizar o número de acidentes, então usamos negativo

    def __hash__(self):
        if self._hash is None:
            condition_str = str(self.conditions)
            self._hash = hash((hashlib.md5(condition_str.encode()).hexdigest(),))
        return self._hash

    def __eq__(self, other):
        return isinstance(other, AccidentNode) and self.conditions == other.conditions

    def __str__(self):
        return f"AccidentNode(conditions={self.conditions}, data_shape={self.data.shape})"

    def __repr__(self):
        return f"AccidentNode(conditions={self.conditions}, data_shape={self.data.shape})"


##==================MAIN====================


# Carregar os dados de acidentes
data = pd.read_csv('../data/acidentes.csv', delimiter=',', encoding='ISO-8859-1', low_memory=False)

data = limpar_dados(data)

# Adicionar uma coluna para contar os acidentes
data['NUM_ACIDENTES'] = 1 
# Pré-processamento para adicionar uma coluna de horário (por exemplo, dia/noite)
data['HORARIO'] = data['DATA HORA_BOLETIM'].apply(lambda x: 'NOITE' if int(x.split()[1].split(':')[0]) >= 18 else 'DIA')


# Inicializar o nó raiz
root = AccidentNode(data)

# Inicializar e executar o MCTS
tree = MCTS()

numero_simulacoes = int(input("Digite o número de simulações: "))


for i in range(numero_simulacoes):  # Ajuste o número de iterações conforme necessário
    tree.do_rollout(root)
    progress = (i + 1) / numero_simulacoes * 100
    sys.stdout.write(f"\rProgresso: {progress:.2f}%")
    sys.stdout.flush()

print("\n")

# Escolher o melhor subgrupo
best_node = tree.choose(root)
print(f"100% concluído")
print(f"Melhor subgrupo de acidentes: {best_node.conditions}")
