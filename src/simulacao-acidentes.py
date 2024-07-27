import pandas as pd
from random import choice
from mcts import MCTS, Node
import hashlib

class AccidentNode(Node):
    def __init__(self, data, conditions=None):
        self.data = data
        self.conditions = conditions if conditions else []
        self.children = None
        self.terminal = False  # Pode ser ajustado para definir estados terminais
        self._hash = None  # Cache do hash

    def find_children(self):
        if self.terminal:
            return set()
        if self.children is None:
            self.children = set()
            possible_conditions = [
                ('TIPO_ACIDENTE', 'COLISÃO'),
                ('DESC_TIPO_ACIDENTE', 'COLISÃO COM MOTOCICLETA'),
                ('DESC_TEMPO', 'CHUVOSO'),
                ('PAVIMENTO', 'MOLHADO'),
                ('HORARIO', 'NOITE')
            ]
            for condition in possible_conditions:
                new_conditions = self.conditions + [condition]
                new_data = self.apply_conditions(self.data, new_conditions)
                self.children.add(AccidentNode(new_data, new_conditions))
        return self.children

    def find_random_child(self):
        children = list(self.find_children())
        if not children:
            return None
        return choice(children)

    def is_terminal(self):
        return self.terminal

    def reward(self):
        if not self.terminal:
            raise RuntimeError("reward called on nonterminal node")
        return self.calculate_reward(self.data)

    def apply_conditions(self, data, conditions):
        filtered_data = data.copy(deep=False)  # Use cópia rasa para eficiência
        for column, value in conditions:
            filtered_data = filtered_data[filtered_data[column] == value]
        return filtered_data

    def calculate_reward(self, data):
        return -data['NUM_ACIDENTES'].sum()  # Queremos minimizar o número de acidentes, então usamos negativo

    def __hash__(self):
        if self._hash is None:
            condition_str = str(self.conditions)
            self._hash = hash((hashlib.md5(condition_str.encode()).hexdigest(),))
        return self._hash

    def __eq__(self, other):
        return isinstance(other, AccidentNode) and self.conditions == other.conditions

# Carregar os dados de acidentes
data = pd.read_csv('../data/acidentes.csv', delimiter=',', encoding='ISO-8859-1', low_memory=False)
data['NUM_ACIDENTES'] = 1  # Adicionar uma coluna para contar os acidentes (para fins de exemplo)

# Pré-processamento para adicionar uma coluna de horário (por exemplo, dia/noite)
data['HORARIO'] = data['DATA HORA_BOLETIM'].apply(lambda x: 'NOITE' if int(x.split()[1].split(':')[0]) >= 18 else 'DIA')

# Inicializar o nó raiz
root = AccidentNode(data)

print("Iniciando simulação de acidentes...")
# Inicializar e executar o MCTS
tree = MCTS()

for i in range(100):  # Ajuste o número de iterações conforme necessário
    if i % 10 == 0:
        print(f"{i}% concluído")
    tree.do_rollout(root)

# Escolher o melhor subgrupo
best_node = tree.choose(root)
print(f"100% concluído")
print(f"Melhor subgrupo de acidentes: {best_node.conditions}")
