import pytest
from networkx import MultiDiGraph
from pyformlang.cfg import CFG, Terminal, Variable

from project.cfpq import cfg_to_weak_normal_form, hellings_based_cfpq


def build_graph(edges: list[tuple]) -> MultiDiGraph:
    graph = MultiDiGraph()
    for source, target, label in edges:
        graph.add_edge(source, target, label=label)
    return graph


GRAMMARS = [
    "S -> $",
    "S -> a S b S | $",
    # ε выводится только из нестартового A
    """
    S -> A b
    A -> a A | $
    """,
    "S -> S S | a | $",
]


@pytest.mark.parametrize("grammar_text", GRAMMARS)
def test_weak_normal_form_shape(grammar_text):
    wcnf = cfg_to_weak_normal_form(CFG.from_text(grammar_text))
    for production in wcnf.productions:
        body = production.body
        assert (
            len(body) == 0
            or (len(body) == 1 and isinstance(body[0], Terminal))
            or (len(body) == 2 and all(isinstance(part, Variable) for part in body))
        ), production


@pytest.mark.parametrize("grammar_text", GRAMMARS)
def test_weak_normal_form_keeps_language(grammar_text):
    cfg = CFG.from_text(grammar_text)
    wcnf = cfg_to_weak_normal_form(cfg)
    for word in ["", "a", "b", "ab", "aab", "abab", "aabb", "ba", "aaa"]:
        assert cfg.contains(word) == wcnf.contains(word), word


def test_hellings_epsilon_gives_loops_on_every_node():
    graph = MultiDiGraph()
    graph.add_nodes_from([0, 1, 2])
    assert hellings_based_cfpq(CFG.from_text("S -> $"), graph) == {
        (0, 0),
        (1, 1),
        (2, 2),
    }


def test_hellings_epsilon_label_is_not_empty_word():
    # ребро "epsilon" не пустой переход: ε-грамматика даёт только петли
    graph = build_graph([(0, 1, "epsilon")])
    assert hellings_based_cfpq(CFG.from_text("S -> $"), graph) == {(0, 0), (1, 1)}
