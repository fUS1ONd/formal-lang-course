from collections import defaultdict

import networkx as nx
from pyformlang.cfg import CFG, Epsilon, Production, Terminal

from project.finite_automata import graph_to_nfa


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """Приводит КС грамматику к ослабленной нормальной форме Хомского.

    Нормальная форма pyformlang удаляет все ε-правила, поэтому возвращаем
    правило A -> ε для каждого нетерминала, из которого выводится ε.
    """
    normal_form = cfg.to_normal_form()
    epsilon_productions = {
        Production(variable, [Epsilon()]) for variable in cfg.get_nullable_symbols()
    }
    return CFG(
        start_symbol=cfg.start_symbol,
        productions=normal_form.productions | epsilon_productions,
    )


def hellings_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """Пары (стартовая, финальная) вершин, между которыми есть путь,
    метки которого образуют слово из языка грамматики."""
    nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    wcnf = cfg_to_weak_normal_form(cfg)

    epsilon_heads = set()
    heads_by_terminal = defaultdict(set)
    heads_by_body = defaultdict(set)
    for production in wcnf.productions:
        match production.body:
            case []:
                epsilon_heads.add(production.head)
            case [Terminal() as terminal]:
                heads_by_terminal[terminal.value].add(production.head)
            case [left, right]:
                heads_by_body[(left, right)].add(production.head)

    # тройка (A, u, v): из A выводится слово на пути из u в v
    triples = {
        (head, state.value, state.value)
        for head in epsilon_heads
        for state in nfa.states
    }
    for source, symbol, target in nfa:
        for head in heads_by_terminal[symbol.value]:
            triples.add((head, source.value, target.value))

    # по вершине начала -> (нетерминал, конец); по вершине конца -> (нетерминал, начало)
    by_start, by_end = defaultdict(set), defaultdict(set)
    for head, start, end in triples:
        by_start[start].add((head, end))
        by_end[end].add((head, start))

    worklist = list(triples)
    while worklist:
        current_head, start, end = worklist.pop()
        new_triples = set()
        for left_head, left_start in by_end[start]:
            for head in heads_by_body[(left_head, current_head)]:
                new_triples.add((head, left_start, end))
        for right_head, right_end in by_start[end]:
            for head in heads_by_body[(current_head, right_head)]:
                new_triples.add((head, start, right_end))
        for triple in new_triples - triples:
            head, triple_start, triple_end = triple
            triples.add(triple)
            by_start[triple_start].add((head, triple_end))
            by_end[triple_end].add((head, triple_start))
            worklist.append(triple)

    start_values = {state.value for state in nfa.start_states}
    final_values = {state.value for state in nfa.final_states}
    return {
        (start, end)
        for head, start, end in triples
        if head == wcnf.start_symbol and start in start_values and end in final_values
    }
