# Tutorial do Algoritmo Monte Carlo Tree Search (MCTS) e sua Explicação com Código Python

## Introdução

Neste tutorial, explicaremos o algoritmo Monte Carlo Tree Search (MCTS) e cada parte do código. Recentemente, aplicamos MCTS para desenvolver nosso jogo.

O código é genérico e assume apenas familiaridade com Python básico. Explicamos em relação ao nosso jogo. Se você quiser usá-lo para seu projeto ou jogo, terá que modificar ligeiramente as funções que mencionei abaixo.


## Regras do Jogo (Modo 1)

1. O jogo é jogado em uma grade 9x9 como Sudoku.
2. Esta grande grade 9x9 é dividida em 9 grades menores 3x3 (tabuleiro local).
3. O objetivo do jogo é ganhar qualquer um dos tabuleiros locais dos 9 disponíveis.
4. Seu movimento determina em qual tabuleiro local a IA deve fazer um movimento e vice-versa.
5. Por exemplo, você faz um movimento na posição 1 do tabuleiro local número 5. Isso forçará a IA a fazer um movimento no tabuleiro local número 1.
6. As regras do Jogo da Velha normal são aplicadas ao tabuleiro local.

## Por que MCTS?

Como você deve ter visto, este jogo tem um fator de ramificação muito alto. Para o primeiro movimento, todo o tabuleiro está vazio. Portanto, há 81 espaços vazios. No primeiro turno, há 81 possíveis movimentos. No segundo turno, aplicando a regra 4, há 8 ou 9 possíveis movimentos.

Para os primeiros 2 movimentos, isso resulta em 81\*9 = 729 combinações possíveis. Assim, o número de combinações possíveis aumenta conforme o jogo progride, resultando em um alto fator de ramificação. Para jogos com um fator de ramificação tão alto, não é possível aplicar o algoritmo minimax. O algoritmo MCTS funciona para esses tipos de jogos.

Além disso, como você deve ter visto jogando o jogo, o tempo que leva para a IA fazer um movimento é de cerca de um segundo. Assim, o MCTS é rápido. O MCTS foi aplicado a ambos os modos do jogo.

## Passos do MCTS

O MCTS consiste em 4 etapas:

1. **Seleção**
2. **Expansão**
3. **Simulação**
4. **Retropropagação**

### Seleção

A ideia é continuar selecionando os melhores nós filhos até alcançarmos o nó folha da árvore. Uma boa maneira de selecionar tal nó filho é usar a fórmula UCT (Upper Confidence Bound applied to trees):

```
wi/ni + c*sqrt(t)/ni
```

- **wi**: número de vitórias após o i-ésimo movimento
- **ni**: número de simulações após o i-ésimo movimento
- **c**: parâmetro de exploração (teoricamente igual a √2)
- **t**: número total de simulações para o nó pai

### Expansão

Quando não pode mais aplicar UCT para encontrar o nó sucessor, ele expande a árvore de jogo anexando todos os estados possíveis a partir do nó folha.

### Simulação

Após a Expansão, o algoritmo escolhe um nó filho arbitrariamente e simula todo o jogo a partir do nó selecionado até alcançar o estado resultante do jogo. Se os nós forem escolhidos aleatoriamente durante a simulação, é chamada de simulação leve. Você também pode optar por uma simulação pesada escrevendo heurísticas de qualidade ou funções de avaliação.

### Retropropagação

Uma vez que o algoritmo alcança o final do jogo, ele avalia o estado para determinar qual jogador venceu. Ele percorre para cima até a raiz e incrementa a pontuação de visita para todos os nós visitados. Também atualiza a pontuação de vitória para cada nó se o jogador daquela posição venceu a simulação.

## Código MCTS em Python

Primeiro, precisamos importar `numpy` e `defaultdict`:

```python
import numpy as np
from collections import defaultdict
```

### Definição da Classe MCTS

```python
class MonteCarloTreeSearchNode():
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = self.untried_actions()
        return
```

#### Variáveis Inicializadas no Construtor

- **state**: Representa o estado do tabuleiro. Geralmente representado por um array. Para Jogo da Velha normal, é um array 3x3.
- **parent**: É None para o nó raiz e para outros nós é igual ao nó de onde foi derivado.
- **children**: Contém todas as ações possíveis a partir do nó atual.
- **parent_action**: None para o nó raiz e para outros nós é igual à ação que seu pai executou.
- **_number_of_visits**: Número de vezes que o nó atual é visitado.
- **_results**: É um dicionário.
- **_untried_actions**: Representa a lista de todas as ações possíveis.
- **action**: Movimento que deve ser executado.

### Funções Membros

Todas as funções abaixo são funções membros, exceto `main()`.

```python
def untried_actions(self):
    self._untried_actions = self.state.get_legal_actions()
    return self._untried_actions
```

Retorna a lista de ações não tentadas a partir de um estado dado. Para o primeiro turno do nosso jogo, há 81 possíveis ações. Para o segundo turno, são 8 ou 9.

```python
def q(self):
    wins = self._results[1]
    loses = self._results[-1]
    return wins - loses
```

Retorna a diferença entre vitórias e derrotas.

```python
def n(self):
    return self._number_of_visits
```

Retorna o número de vezes que cada nó é visitado.

```python
def expand(self):
    action = self._untried_actions.pop()
    next_state = self.state.move(action)
    child_node = MonteCarloTreeSearchNode(next_state, parent=self, parent_action=action)
    self.children.append(child_node)
    return child_node
```

A partir do estado atual, o próximo estado é gerado dependendo da ação executada. Neste passo, todos os possíveis nós filhos correspondentes aos estados gerados são anexados ao array `children` e o `child_node` é retornado.

```python
def is_terminal_node(self):
    return self.state.is_game_over()
```

Usado para verificar se o nó atual é terminal ou não. O nó terminal é alcançado quando o jogo termina.

```python
def rollout(self):
    current_rollout_state = self.state
    
    while not current_rollout_state.is_game_over():
        possible_moves = current_rollout_state.get_legal_actions()
        action = self.rollout_policy(possible_moves)
        current_rollout_state = current_rollout_state.move(action)
    return current_rollout_state.game_result()
```

A partir do estado atual, o jogo inteiro é simulado até que haja um resultado. Esse resultado é retornado. Se resultar em uma vitória, o resultado é 1. Caso contrário, é -1 se resultar em uma derrota. E é 0 se for empate.

```python
def backpropagate(self, result):
    self._number_of_visits += 1
    self._results[result] += 1
    if self.parent:
        self.parent.backpropagate(result)
```

Após alcançar o final do jogo, o algoritmo percorre para cima até a raiz e incrementa a pontuação de visita para todos os nós visitados, atualizando a pontuação de vitória para cada nó se o jogador daquela posição venceu a simulação.
