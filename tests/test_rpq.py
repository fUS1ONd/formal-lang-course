import random

import cfpq_data
import pytest
from networkx import MultiDiGraph

from project.finite_automata import graph_to_nfa, regex_to_dfa
from project.rpq import tensor_based_rpq


def build_graph(edges: list[tuple]) -> MultiDiGraph:
    graph = MultiDiGraph()
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph


def naive_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    """Эталон: для каждой пары вершин отдельное пересечение средствами pyformlang."""
    dfa = regex_to_dfa(regex)
    return {
        (start, final)
        for start in start_nodes
        for final in final_nodes
        if not dfa.get_intersection(graph_to_nfa(graph, {start}, {final})).is_empty()
    }


def test_empty_word_gives_pairs_of_same_node():
    graph = build_graph([(0, 1, "a")])

    assert tensor_based_rpq("b*", graph, {0, 1}, {0, 1}) == {(0, 0), (1, 1)}


def test_only_given_start_and_final_nodes():
    graph = build_graph([(0, 1, "a"), (1, 2, "a"), (2, 0, "a")])

    assert tensor_based_rpq("a*", graph, {0}, {2}) == {(0, 2)}
    assert tensor_based_rpq("a a", graph, {1}, {0, 2}) == {(1, 0)}


def test_empty_node_sets_mean_all_nodes():
    graph = build_graph([(0, 1, "a"), (1, 2, "b")])

    assert tensor_based_rpq("a b", graph, set(), set()) == {(0, 2)}


def test_label_absent_in_graph():
    graph = build_graph([(0, 1, "a")])

    assert tensor_based_rpq("c", graph, {0, 1}, {0, 1}) == set()


def test_epsilon_like_label_is_regular_symbol():
    graph = build_graph([(0, 1, "epsilon")])

    assert tensor_based_rpq("a*", graph, {0}, {1}) == set()


def test_unknown_node_raises():
    graph = build_graph([(0, 1, "a")])

    with pytest.raises(ValueError):
        tensor_based_rpq("a", graph, {42}, {1})


def test_two_cycles_graph():
    # циклы 0-1-2-3-0 по "a" и 0-4-5-0 по "b"
    graph = cfpq_data.labeled_two_cycles_graph(3, 2, labels=("a", "b"))

    assert tensor_based_rpq("a a a a", graph, {0}, set(graph.nodes)) == {(0, 0)}
    assert tensor_based_rpq("a* b", graph, {1}, set(graph.nodes)) == {(1, 4)}


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("regex", ["a", "a*", "(a|b)* c", "a b* | c", "(a b)* c*"])
def test_matches_naive_on_random_graphs(seed, regex):
    random.seed(seed)
    graph = cfpq_data.labeled_binomial_graph(8, 0.3, labels=["a", "b", "c"], seed=seed)
    nodes = list(graph.nodes)
    start_nodes = set(random.sample(nodes, random.randint(1, len(nodes))))
    final_nodes = set(random.sample(nodes, random.randint(1, len(nodes))))

    assert tensor_based_rpq(regex, graph, start_nodes, final_nodes) == naive_rpq(
        regex, graph, start_nodes, final_nodes
    )
