from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)
from pyformlang.regular_expression import Regex


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    """Строит минимальный ДКА по регулярному выражению в синтаксисе pyformlang."""
    return Regex(regex).to_epsilon_nfa().to_deterministic().minimize()


def graph_to_nfa(
    graph: MultiDiGraph, start_states: set[int], final_states: set[int]
) -> NondeterministicFiniteAutomaton:
    """Строит НКА по графу: вершины - состояния, помеченные рёбра - переходы.

    Если множество стартовых или финальных вершин пусто, такими считаются все вершины.
    """
    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transitions(
        [
            (source, label, target)
            for source, target, label in graph.edges(data="label")
            if label is not None
        ]
    )

    for node in start_states or graph.nodes:
        nfa.add_start_state(node)
    for node in final_states or graph.nodes:
        nfa.add_final_state(node)

    return nfa
