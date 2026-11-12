from collections import defaultdict

import networkx as nx
from pyformlang.cfg import CFG, Epsilon, Production, Terminal, Variable
from scipy.sparse import coo_array, csr_array

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


def _split_productions(
    wcnf: CFG,
) -> tuple[
    set[Variable],
    dict[str, set[Variable]],
    dict[tuple[Variable, Variable], set[Variable]],
]:
    """Раскладывает продукции ОНФХ по видам: A -> ε, A -> a и A -> B C."""
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
    return epsilon_heads, heads_by_terminal, heads_by_body


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
    epsilon_heads, heads_by_terminal, heads_by_body = _split_productions(wcnf)

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


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """То же, что hellings_based_cfpq, но через булевы матрицы.

    Для каждого нетерминала A матрица |V| x |V|: единица в (u, v) значит,
    что из A выводится слово на пути из u в v. Для каждого A -> B C
    добавляем M_B @ M_C к M_A, пока матрицы растут.
    """
    nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    wcnf = cfg_to_weak_normal_form(cfg)
    epsilon_heads, heads_by_terminal, heads_by_body = _split_productions(wcnf)

    states = list(nfa.states)
    state_to_index = {state: index for index, state in enumerate(states)}
    states_count = len(states)

    # начальное заполнение: A -> ε даёт петли, A -> a даёт рёбра с меткой a
    cells = defaultdict(list)
    for head in epsilon_heads:
        cells[head].extend((index, index) for index in range(states_count))
    for source, symbol, target in nfa:
        for head in heads_by_terminal[symbol.value]:
            cells[head].append((state_to_index[source], state_to_index[target]))

    variables = wcnf.variables | {wcnf.start_symbol}
    matrices: dict[Variable, csr_array] = {}
    for variable in variables:
        rows, columns = zip(*cells[variable]) if cells[variable] else ((), ())
        matrices[variable] = coo_array(
            ([True] * len(rows), (rows, columns)),
            shape=(states_count, states_count),
            dtype=bool,
        ).tocsr()

    binary_productions = [
        (head, left, right)
        for (left, right), heads in heads_by_body.items()
        for head in heads
    ]
    changed = True
    while changed:
        changed = False
        for head, left, right in binary_productions:
            updated = matrices[head] + matrices[left] @ matrices[right]
            if updated.nnz != matrices[head].nnz:
                matrices[head] = updated
                changed = True

    start_indices = {state_to_index[state] for state in nfa.start_states}
    final_indices = {state_to_index[state] for state in nfa.final_states}
    return {
        (states[row].value, states[column].value)
        for row, column in zip(*matrices[wcnf.start_symbol].nonzero())
        if row in start_indices and column in final_indices
    }
