from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    Symbol,
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
    Вершины, которых нет в графе, приводят к ValueError.
    """
    unknown_nodes = (set(start_states or ()) | set(final_states or ())) - set(
        graph.nodes
    )
    if unknown_nodes:
        raise ValueError(f"вершин нет в графе: {sorted(unknown_nodes, key=str)}")

    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transitions(
        [
            # Symbol не даёт pyformlang принять метку "epsilon" за пустое слово
            (source, Symbol(label), target)
            for source, target, label in graph.edges(data="label")
            if label is not None
        ]
    )

    for node in start_states or graph.nodes:
        nfa.add_start_state(node)
    for node in final_states or graph.nodes:
        nfa.add_final_state(node)

    return nfa
