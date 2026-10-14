import random

import cfpq_data
import pytest
from networkx import MultiDiGraph

from project.finite_automata import graph_to_nfa, regex_to_dfa
from project.rpq import ms_bfs_based_rpq, tensor_based_rpq


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


@pytest.fixture(params=[tensor_based_rpq, ms_bfs_based_rpq])
def rpq(request):
    return request.param


def test_empty_word_gives_pairs_of_same_node(rpq):
    graph = build_graph([(0, 1, "a")])

    assert rpq("b*", graph, {0, 1}, {0, 1}) == {(0, 0), (1, 1)}


def test_only_given_start_and_final_nodes(rpq):
    graph = build_graph([(0, 1, "a"), (1, 2, "a"), (2, 0, "a")])

    assert rpq("a*", graph, {0}, {2}) == {(0, 2)}
    assert rpq("a a", graph, {1}, {0, 2}) == {(1, 0)}


def test_empty_node_sets_mean_all_nodes(rpq):
    graph = build_graph([(0, 1, "a"), (1, 2, "b")])

    assert rpq("a b", graph, set(), set()) == {(0, 2)}


def test_label_absent_in_graph(rpq):
    graph = build_graph([(0, 1, "a")])

    assert rpq("c", graph, {0, 1}, {0, 1}) == set()


def test_epsilon_like_label_is_regular_symbol(rpq):
    graph = build_graph([(0, 1, "epsilon")])

    assert rpq("a*", graph, {0}, {1}) == set()


def test_unknown_node_raises(rpq):
    graph = build_graph([(0, 1, "a")])

    with pytest.raises(ValueError):
        rpq("a", graph, {42}, {1})


def test_empty_regex_language(rpq):
    graph = build_graph([(0, 1, "a")])

    # пустая строка в pyformlang - пустой язык ("$" - это пустое слово)
    assert rpq("", graph, {0, 1}, {0, 1}) == set()


def test_start_nodes_do_not_share_progress(rpq):
    # из 0 до 2 идёт слово "a b", из 1 - только "b", поэтому (1, 2) не ответ на "a b"
    graph = build_graph([(0, 1, "a"), (1, 2, "b")])

    assert rpq("a b", graph, {0, 1}, {2}) == {(0, 2)}


def test_two_cycles_graph(rpq):
    # циклы 0-1-2-3-0 по "a" и 0-4-5-0 по "b"
    graph = cfpq_data.labeled_two_cycles_graph(3, 2, labels=("a", "b"))

    assert rpq("a a a a", graph, {0}, set(graph.nodes)) == {(0, 0)}
    assert rpq("a* b", graph, {1}, set(graph.nodes)) == {(1, 4)}


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("regex", ["a", "a*", "(a|b)* c", "a b* | c", "(a b)* c*"])
def test_matches_naive_on_random_graphs(rpq, seed, regex):
    random.seed(seed)
    graph = cfpq_data.labeled_binomial_graph(8, 0.3, labels=["a", "b", "c"], seed=seed)
    nodes = list(graph.nodes)
    start_nodes = set(random.sample(nodes, random.randint(1, len(nodes))))
    final_nodes = set(random.sample(nodes, random.randint(1, len(nodes))))

    assert rpq(regex, graph, start_nodes, final_nodes) == naive_rpq(
        regex, graph, start_nodes, final_nodes
    )
