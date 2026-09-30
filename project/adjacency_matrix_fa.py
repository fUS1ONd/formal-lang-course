from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from scipy.sparse import coo_array, csr_array, eye_array, kron


class AdjacencyMatrixFA:
    """Конечный автомат как булева декомпозиция матрицы смежности:
    для каждого символа своя разреженная матрица переходов между состояниями.
    """

    def __init__(self, automaton: NondeterministicFiniteAutomaton):
        self.states: list[State] = list(automaton.states)
        self.state_to_index: dict[State, int] = {
            state: index for index, state in enumerate(self.states)
        }
        self.start_indices: set[int] = {
            self.state_to_index[state] for state in automaton.start_states
        }
        self.final_indices: set[int] = {
            self.state_to_index[state] for state in automaton.final_states
        }

        transitions: dict[Symbol, list[tuple[int, int]]] = defaultdict(list)
        for source, symbol, target in automaton:
            transitions[symbol].append(
                (self.state_to_index[source], self.state_to_index[target])
            )

        states_count = len(self.states)
        self.matrices: dict[Symbol, csr_array] = {}
        for symbol, edges in transitions.items():
            sources, targets = zip(*edges)
            self.matrices[symbol] = coo_array(
                ([True] * len(edges), (sources, targets)),
                shape=(states_count, states_count),
                dtype=bool,
            ).tocsr()

    @classmethod
    def from_matrices(
        cls,
        states: list[State],
        start_indices: set[int],
        final_indices: set[int],
        matrices: dict[Symbol, csr_array],
    ) -> "AdjacencyMatrixFA":
        automaton = cls.__new__(cls)
        automaton.states = states
        automaton.state_to_index = {state: index for index, state in enumerate(states)}
        automaton.start_indices = start_indices
        automaton.final_indices = final_indices
        automaton.matrices = matrices
        return automaton

    def accepts(self, word: Iterable[Symbol]) -> bool:
        current = np.zeros(len(self.states), dtype=bool)
        current[list(self.start_indices)] = True
        for symbol in word:
            matrix = self.matrices.get(symbol)
            if matrix is None:
                return False
            current = current @ matrix
        return bool(current[list(self.final_indices)].any())

    def transitive_closure(self) -> csr_array:
        """Рефлексивно-транзитивное замыкание: (i, j) достижимо, если есть путь из i в j,
        включая пустой путь."""
        closure = eye_array(len(self.states), dtype=bool, format="csr")
        for matrix in self.matrices.values():
            closure = closure + matrix
        while True:
            next_closure = closure @ closure
            if next_closure.nnz == closure.nnz:
                return closure
            closure = next_closure

    def is_empty(self) -> bool:
        closure = self.transitive_closure()
        return not closure[list(self.start_indices)][:, list(self.final_indices)].nnz


def intersect_automata(
    automaton1: AdjacencyMatrixFA, automaton2: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    """Пересечение через тензорное (кронекерово) произведение матриц общих символов."""
    second_count = len(automaton2.states)

    # kron кладёт пару (i, j) в индекс i * second_count + j, в том же порядке строим состояния
    def pair_index(first: int, second: int) -> int:
        return first * second_count + second

    states = [
        State((first, second))
        for first in automaton1.states
        for second in automaton2.states
    ]
    start_indices = {
        pair_index(first, second)
        for first in automaton1.start_indices
        for second in automaton2.start_indices
    }
    final_indices = {
        pair_index(first, second)
        for first in automaton1.final_indices
        for second in automaton2.final_indices
    }
    matrices = {
        symbol: kron(automaton1.matrices[symbol], automaton2.matrices[symbol], "csr")
        for symbol in automaton1.matrices.keys() & automaton2.matrices.keys()
    }
    return AdjacencyMatrixFA.from_matrices(
        states, start_indices, final_indices, matrices
    )
