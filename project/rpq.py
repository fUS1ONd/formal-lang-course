from networkx import MultiDiGraph
from scipy.sparse import coo_array, csr_array, eye_array, kron

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


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    """То же, что tensor_based_rpq, но через BFS сразу из всех стартовых вершин.

    Фронт - вертикально сложенные блоки |Q_regex| x |V|, по блоку на стартовую вершину:
    единица в (q, v) блока s значит, что из s дошли до v, а ДКА при этом в состоянии q.
    """
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    regex_fa = AdjacencyMatrixFA(regex_to_dfa(regex))

    graph_starts = list(graph_fa.start_indices)
    regex_start = next(iter(regex_fa.start_indices))
    regex_count = len(regex_fa.states)
    graph_count = len(graph_fa.states)

    front = coo_array(
        (
            [True] * len(graph_starts),
            (
                [
                    block * regex_count + regex_start
                    for block in range(len(graph_starts))
                ],
                graph_starts,
            ),
        ),
        shape=(len(graph_starts) * regex_count, graph_count),
        dtype=bool,
    ).tocsr()

    # D_a^T для каждого блока сразу: блочно-диагональная матрица
    blocks_identity = eye_array(len(graph_starts), dtype=bool, format="csr")
    steps = [
        (kron(blocks_identity, regex_fa.matrices[symbol].T, "csr"), graph_matrix)
        for symbol, graph_matrix in graph_fa.matrices.items()
        if symbol in regex_fa.matrices
    ]

    visited = front
    while front.nnz:
        next_front = csr_array(front.shape, dtype=bool)
        for regex_step, graph_step in steps:
            next_front = next_front + regex_step @ (front @ graph_step)
        front = next_front > visited
        visited = visited + front

    regex_finals = regex_fa.final_indices
    graph_finals = graph_fa.final_indices
    result = set()
    for row, column in zip(*visited.nonzero()):
        block, regex_state = divmod(row, regex_count)
        if regex_state in regex_finals and column in graph_finals:
            graph_start = graph_fa.states[graph_starts[block]]
            result.add((graph_start.value, graph_fa.states[column].value))
    return result
