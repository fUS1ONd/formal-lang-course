from networkx import MultiDiGraph

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.finite_automata import graph_to_nfa, regex_to_dfa


def tensor_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    """Пары (стартовая, финальная) вершин графа, связанные путём,
    метки которого образуют слово из языка регулярного выражения."""
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))
    intersection = intersect_automata(graph_fa, regex_fa)

    start_indices = list(intersection.start_indices)
    final_indices = list(intersection.final_indices)
    reachable = intersection.transitive_closure()[start_indices][:, final_indices]

    result = set()
    for row, column in zip(*reachable.nonzero()):
        graph_start, _ = intersection.states[start_indices[row]].value
        graph_final, _ = intersection.states[final_indices[column]].value
        result.add((graph_start.value, graph_final.value))
    return result
