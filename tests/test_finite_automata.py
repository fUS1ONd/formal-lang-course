import cfpq_data
import pytest
from networkx import MultiDiGraph

from project.finite_automata import graph_to_nfa, regex_to_dfa


@pytest.mark.parametrize(
    "regex",
    ["a", "a b c", "a*", "(a|b)* c", "(a b) | (a c)", "a* a* b", "$"],
)
def test_regex_to_dfa_is_minimal_dfa(regex):
    dfa = regex_to_dfa(regex)

    assert dfa.is_deterministic()
    # повторная минимизация не уменьшает число состояний
    assert len(dfa.minimize().states) == len(dfa.states)


@pytest.mark.parametrize(
    "regex, word",
    [
        ("a", ["a"]),
        ("a b c", ["a", "b", "c"]),
        ("a*", []),
        ("a*", ["a", "a", "a"]),
        ("(a|b)* c", ["c"]),
        ("(a|b)* c", ["a", "b", "a", "c"]),
        ("(a b) | (a c)", ["a", "c"]),
        ("$", []),
    ],
)
def test_regex_to_dfa_accepts(regex, word):
    assert regex_to_dfa(regex).accepts(word)


@pytest.mark.parametrize(
    "regex, word",
    [
        ("a", []),
        ("a", ["b"]),
        ("a b c", ["a", "c"]),
        ("a*", ["b"]),
        ("(a|b)* c", ["a", "b"]),
        ("(a b) | (a c)", ["a", "b", "c"]),
        ("$", ["a"]),
    ],
)
def test_regex_to_dfa_rejects(regex, word):
    assert not regex_to_dfa(regex).accepts(word)


def test_regex_to_dfa_empty_regex_gives_empty_language():
    assert regex_to_dfa("").is_empty()


def test_regex_to_dfa_symbol_is_whole_token():
    # символы разделяются пробелами, поэтому "abc" --- один символ
    dfa = regex_to_dfa("abc | d")

    assert dfa.accepts(["abc"])
    assert not dfa.accepts(["a", "b", "c"])


def build_graph(edges: list[tuple[int, int, str]]) -> MultiDiGraph:
    graph = MultiDiGraph()
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph


def test_graph_to_nfa_uses_explicit_start_and_final():
    graph = build_graph([(0, 1, "a"), (1, 2, "b"), (2, 0, "c")])

    nfa = graph_to_nfa(graph, {0}, {2})

    assert nfa.accepts(["a", "b"])
    assert not nfa.accepts(["a"])
    assert not nfa.accepts(["a", "b", "c"])


def test_graph_to_nfa_empty_sets_mean_all_nodes():
    graph = build_graph([(0, 1, "a"), (1, 2, "b")])

    nfa = graph_to_nfa(graph, set(), set())

    assert nfa.start_states == set(nfa.states) == nfa.final_states
    # любой путь графа, включая пустой, теперь принимается
    assert nfa.accepts([])
    assert nfa.accepts(["b"])
    assert nfa.accepts(["a", "b"])
    assert not nfa.accepts(["b", "a"])


def test_graph_to_nfa_keeps_cycles():
    graph = cfpq_data.labeled_two_cycles_graph(2, 2, labels=("a", "b"))

    nfa = graph_to_nfa(graph, {0}, {0})

    assert nfa.accepts(["a"] * 3)
    assert nfa.accepts(["b"] * 3)
    assert nfa.accepts(["a", "a", "a", "b", "b", "b"])
    assert not nfa.accepts(["a", "b"])


def test_graph_to_nfa_handles_parallel_edges():
    graph = build_graph([(0, 1, "a"), (0, 1, "a"), (0, 1, "b")])

    nfa = graph_to_nfa(graph, {0}, {1})

    assert nfa.accepts(["a"])
    assert nfa.accepts(["b"])


def test_graph_to_nfa_skips_edges_without_label():
    graph = build_graph([(0, 1, "a")])
    graph.add_edge(1, 2)

    nfa = graph_to_nfa(graph, {0}, {2})

    assert nfa.is_empty()


def test_graph_to_nfa_empty_graph():
    assert graph_to_nfa(MultiDiGraph(), set(), set()).is_empty()
