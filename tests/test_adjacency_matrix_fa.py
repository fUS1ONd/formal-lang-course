import pytest
from networkx import MultiDiGraph
from pyformlang.finite_automaton import Symbol

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
from project.finite_automata import graph_to_nfa, regex_to_dfa


def build_graph(edges: list[tuple]) -> MultiDiGraph:
    graph = MultiDiGraph()
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph


def from_regex(regex: str) -> AdjacencyMatrixFA:
    return AdjacencyMatrixFA(regex_to_dfa(regex))


@pytest.mark.parametrize(
    "regex, word, expected",
    [
        ("a b*", "a", True),
        ("a b*", "abbb", True),
        ("a b*", "", False),
        ("a b*", "ba", False),
        ("a*", "", True),
        ("(a|b)* c", "abac", True),
        ("(a|b)* c", "ab", False),
        ("a", "x", False),
    ],
)
def test_accepts(regex, word, expected):
    assert from_regex(regex).accepts(word) == expected


def test_accepts_symbols_and_strings_equally():
    automaton = from_regex("a b")

    assert automaton.accepts([Symbol("a"), Symbol("b")])
    assert automaton.accepts(["a", "b"])


def test_accepts_nfa_with_branching():
    graph = build_graph([(0, 1, "a"), (0, 2, "a"), (2, 3, "b")])
    automaton = AdjacencyMatrixFA(graph_to_nfa(graph, {0}, {3}))

    assert automaton.accepts("ab")
    assert not automaton.accepts("a")


def test_parallel_edges_do_not_break_matrix():
    graph = build_graph([(0, 1, "a"), (0, 1, "a")])
    automaton = AdjacencyMatrixFA(graph_to_nfa(graph, {0}, {1}))

    assert automaton.matrices[Symbol("a")].nnz == 1
    assert automaton.accepts("a")


@pytest.mark.parametrize(
    "regex, expected",
    [("", True), ("a", False), ("a*", False), ("(a|b)* c", False)],
)
def test_is_empty_for_regex(regex, expected):
    assert from_regex(regex).is_empty() == expected


def test_is_empty_when_finals_unreachable():
    graph = build_graph([(0, 1, "a"), (2, 3, "b")])

    assert AdjacencyMatrixFA(graph_to_nfa(graph, {0}, {3})).is_empty()
    assert not AdjacencyMatrixFA(graph_to_nfa(graph, {0}, {1})).is_empty()


def test_is_empty_on_long_path():
    graph = build_graph([(node, node + 1, "a") for node in range(100)])

    assert not AdjacencyMatrixFA(graph_to_nfa(graph, {0}, {100})).is_empty()
    assert AdjacencyMatrixFA(graph_to_nfa(graph, {100}, {0})).is_empty()


@pytest.mark.parametrize(
    "regex1, regex2, word, expected",
    [
        ("(a|b)*", "a b*", "abb", True),
        ("(a|b)*", "a b*", "ba", False),
        ("a*", "b*", "", True),
        ("a*", "b*", "a", False),
        ("a b c", "(a|b|c)* c", "abc", True),
    ],
)
def test_intersection_accepts(regex1, regex2, word, expected):
    intersection = intersect_automata(from_regex(regex1), from_regex(regex2))

    assert intersection.accepts(word) == expected


@pytest.mark.parametrize(
    "regex1, regex2, expected",
    [("a", "b", True), ("a b", "a c", True), ("a*", "a", False), ("", "a*", True)],
)
def test_intersection_is_empty(regex1, regex2, expected):
    intersection = intersect_automata(from_regex(regex1), from_regex(regex2))

    assert intersection.is_empty() == expected


def test_intersection_state_pairs_match_kron_order():
    first = from_regex("a b")
    second = from_regex("a*")
    intersection = intersect_automata(first, second)

    for index, state in enumerate(intersection.states):
        first_state, second_state = state.value
        assert index == (
            first.state_to_index[first_state] * len(second.states)
            + second.state_to_index[second_state]
        )
