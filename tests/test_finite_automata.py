import cfpq_data
import networkx as nx
import pytest
from networkx import MultiDiGraph
from pyformlang.regular_expression.regex_objects import MisformedRegexError

from project.finite_automata import graph_to_nfa, regex_to_dfa
from project.graph_utils import save_two_cycles_graph


def build_graph(edges: list[tuple]) -> MultiDiGraph:
    graph = MultiDiGraph()
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph


def values(finite_automaton_objects) -> set:
    return {obj.value for obj in finite_automaton_objects}


@pytest.mark.parametrize(
    "regex",
    ["a", "a b c", "a*", "(a|b)* c", "(a b) | (a c)", "a* a* b", "$"],
)
def test_regex_to_dfa_is_minimal_dfa(regex):
    dfa = regex_to_dfa(regex)

    assert dfa.is_deterministic()
    # повторная минимизация не уменьшает число состояний
    assert len(dfa.minimize().states) == len(dfa.states)


def test_regex_to_dfa_is_minimal_for_all_course_regexes():
    # автотесты преподавателя берут из этой таблицы случайную выборку
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent / "autotests"))
    from constants import REGEXP
    from grammars_constants import GRAMMARS_TABLE

    for regex in (regex for row in GRAMMARS_TABLE for regex in row[REGEXP]):
        dfa = regex_to_dfa(regex)
        minimized = dfa.minimize()

        assert dfa.is_deterministic(), regex
        assert len(minimized.states) == len(dfa.states), regex
        assert dfa.is_equivalent_to(minimized), regex


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


@pytest.mark.parametrize("regex", ["", "   "])
def test_regex_to_dfa_blank_regex_gives_empty_language(regex):
    assert regex_to_dfa(regex).is_empty()


@pytest.mark.parametrize("regex", ["(", ")", "*", "|"])
def test_regex_to_dfa_rejects_malformed_regex(regex):
    with pytest.raises((MisformedRegexError, ValueError)):
        regex_to_dfa(regex)


def test_regex_to_dfa_symbol_is_whole_token():
    # символы разделяются пробелами, поэтому "abc" - один символ
    dfa = regex_to_dfa("abc | d")

    assert dfa.accepts(["abc"])
    assert not dfa.accepts(["a", "b", "c"])


def test_regex_to_dfa_dot_is_concatenation_and_plus_is_union():
    assert regex_to_dfa("a.b").accepts(["a", "b"])
    assert not regex_to_dfa("a.b").accepts(["a.b"])
    assert regex_to_dfa("a+b").accepts(["a"])
    assert not regex_to_dfa("a+b").accepts(["a+b"])


@pytest.mark.parametrize("regex", ["$", "epsilon"])
def test_regex_to_dfa_epsilon_tokens_mean_empty_word(regex):
    dfa = regex_to_dfa(regex)

    assert dfa.accepts([])
    assert not dfa.accepts([regex])


def test_regex_to_dfa_newline_is_part_of_a_symbol():
    # токены разделяются только пробелами, перенос строки в разделители не входит
    assert values(regex_to_dfa("a\nb").symbols) == {"a\nb"}


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


def test_graph_to_nfa_none_is_treated_as_empty_set():
    graph = build_graph([(0, 1, "a")])

    nfa = graph_to_nfa(graph, None, None)

    assert nfa.start_states == nfa.final_states == set(nfa.states)


def test_graph_to_nfa_accepts_node_view_instead_of_set():
    # вершины передают не только множеством, но и NodeView самого графа
    graph = build_graph([(0, 1, "a")])

    nfa = graph_to_nfa(graph, graph.nodes, graph.nodes)

    assert nfa.accepts([])
    assert nfa.accepts(["a"])


@pytest.mark.parametrize("start_states, final_states", [({42}, {1}), ({0}, {99})])
def test_graph_to_nfa_rejects_nodes_missing_in_graph(start_states, final_states):
    graph = build_graph([(0, 1, "a")])

    with pytest.raises(ValueError):
        graph_to_nfa(graph, start_states, final_states)


def test_graph_to_nfa_rejects_int_nodes_of_a_dot_graph(tmp_path):
    # граф задачи 1 попадает к пользователю через DOT-файл, а его вершины - строки
    path = tmp_path / "two_cycles.dot"
    save_two_cycles_graph(2, 2, ("a", "b"), path)
    graph = nx.nx_pydot.read_dot(path)

    with pytest.raises(ValueError):
        graph_to_nfa(graph, {0}, {0})

    # первый цикл проходит через вершины 0, 1, 2, то есть возврат в 0 - это три шага
    assert graph_to_nfa(graph, {"0"}, {"0"}).accepts(["a"] * 3)


def test_graph_to_nfa_keeps_cycles():
    graph = cfpq_data.labeled_two_cycles_graph(2, 2, labels=("a", "b"))

    nfa = graph_to_nfa(graph, {0}, {0})

    assert nfa.accepts(["a"] * 3)
    assert nfa.accepts(["b"] * 3)
    assert nfa.accepts(["a", "a", "a", "b", "b", "b"])
    assert not nfa.accepts(["a", "b"])


def test_graph_to_nfa_self_loop_gives_arbitrary_number_of_repetitions():
    graph = build_graph([(0, 0, "a")])

    nfa = graph_to_nfa(graph, {0}, {0})

    assert nfa.accepts([])
    assert nfa.accepts(["a"] * 7)


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


def test_graph_to_nfa_unlabeled_edge_does_not_drop_the_labeled_ones():
    graph = build_graph([(0, 1, "a"), (1, 2, "b")])
    graph.add_edge(2, 3)

    nfa = graph_to_nfa(graph, {0}, {2})

    assert nfa.accepts(["a", "b"])


@pytest.mark.parametrize("label", [1, 0, "", "x" * 10000, "🙂", "مرحبا"])
def test_graph_to_nfa_uses_unusual_label_as_is(label):
    graph = build_graph([(0, 1, label)])

    nfa = graph_to_nfa(graph, {0}, {1})

    assert values(nfa.symbols) == {label}
    assert nfa.accepts([label])


@pytest.mark.parametrize("label", ["epsilon", "ɛ"])
def test_graph_to_nfa_epsilon_like_label_is_not_a_free_move(label):
    # pyformlang считает эти строки пустым словом, если не обернуть их в Symbol
    graph = build_graph([(0, 1, label), (1, 2, "a")])

    nfa = graph_to_nfa(graph, {0}, {2})

    assert not nfa.accepts(["a"])
    assert not nfa.get_intersection(regex_to_dfa("a")).accepts(["a"])
    assert nfa.accepts([label, "a"]) == nfa.to_deterministic().accepts([label, "a"])


def test_graph_to_nfa_isolated_vertex_is_a_state():
    # из вершины без рёбер есть пустой путь в саму себя, состоянием она быть обязана
    graph = build_graph([(0, 1, "a")])
    graph.add_node(7)

    nfa = graph_to_nfa(graph, set(), set())

    assert 7 in values(nfa.states)
    assert 7 in values(nfa.start_states) and 7 in values(nfa.final_states)


def test_graph_to_nfa_graph_without_edges_accepts_only_empty_word():
    graph = MultiDiGraph()
    graph.add_nodes_from([0, 1, 2])

    nfa = graph_to_nfa(graph, set(), set())

    assert nfa.accepts([])
    assert not nfa.accepts(["a"])


def test_graph_to_nfa_empty_graph():
    assert graph_to_nfa(MultiDiGraph(), set(), set()).is_empty()


def test_graph_to_nfa_vertex_that_is_both_start_and_final_accepts_empty_word():
    graph = build_graph([(0, 1, "a")])

    assert graph_to_nfa(graph, {0}, {0}).accepts([])


def test_graph_to_nfa_unreachable_final_vertex_gives_empty_language():
    graph = build_graph([(0, 1, "a"), (2, 3, "b")])

    assert graph_to_nfa(graph, {0}, {3}).is_empty()


def test_graph_to_nfa_does_not_mutate_the_graph():
    graph = build_graph([(0, 1, "a"), (1, 2, "b")])
    before = (
        sorted(graph.nodes(data=True), key=str),
        sorted(graph.edges(data=True), key=str),
    )

    graph_to_nfa(graph, {0}, {2})

    after = (
        sorted(graph.nodes(data=True), key=str),
        sorted(graph.edges(data=True), key=str),
    )
    assert before == after


def test_graph_to_nfa_does_not_follow_graph_changes_made_after_construction():
    graph = build_graph([(0, 1, "a")])
    nfa = graph_to_nfa(graph, {0}, {1})

    graph.add_edge(1, 2, label="b")

    assert not nfa.accepts(["a", "b"])


def test_graph_to_nfa_is_repeatable():
    graph = build_graph([(0, 1, "a"), (1, 0, "b")])

    assert graph_to_nfa(graph, {0}, {1}).is_equivalent_to(graph_to_nfa(graph, {0}, {1}))


def test_intersection_keeps_only_words_present_in_the_graph():
    graph = build_graph([(0, 1, "a"), (1, 2, "b"), (2, 0, "c")])
    nfa = graph_to_nfa(graph, {0}, {2})

    intersection = nfa.get_intersection(regex_to_dfa("a b | a c"))

    assert intersection.accepts(["a", "b"])
    assert not intersection.accepts(["a", "c"])


def test_intersection_finds_multi_letter_labels_of_real_graphs():
    graph = build_graph([(0, 1, "subClassOf"), (1, 2, "type")])
    nfa = graph_to_nfa(graph, {0}, {2})

    assert not nfa.get_intersection(regex_to_dfa("subClassOf type")).is_empty()
