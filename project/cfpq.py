from collections import defaultdict

import networkx as nx
from pyformlang.cfg import CFG, Epsilon, Production, Terminal, Variable
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State, Symbol
from pyformlang.rsa import Box, RecursiveAutomaton
from scipy.sparse import coo_array, csr_array

from project.adjacency_matrix_fa import AdjacencyMatrixFA, intersect_automata
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


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """Строит RSM по грамматике в EBNF со стартовым нетерминалом S.

    Каждая строка - "A -> регулярное выражение", строки с одной головой
    объединяются через "|".
    """
    return RecursiveAutomaton.from_text(ebnf)


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """Строит RSM по КС грамматике.

    Тело продукции - частный случай регулярного выражения,
    поэтому текст грамматики разбирается как EBNF.
    """
    start_symbol = Symbol(cfg.start_symbol.value)
    if all(production.head != cfg.start_symbol for production in cfg.productions):
        # from_text не создаёт бокс для нетерминала без продукций,
        # поэтому стартовому даём пустой бокс: из него ничего не выводится
        return RecursiveAutomaton(
            initial_label=start_symbol, boxes={Box(label=start_symbol)}
        )
    return RecursiveAutomaton.from_text(cfg.to_text(), start_symbol)


def _rsm_to_adjacency_matrix_fa(
    rsm: RecursiveAutomaton,
) -> tuple[AdjacencyMatrixFA, dict[Symbol, tuple[set[int], set[int]]]]:
    """Склеивает боксы RSM в один автомат с состояниями (нетерминал, состояние бокса).

    Возвращает автомат и для каждого нетерминала индексы стартовых
    и финальных состояний его бокса.
    """
    nfa = NondeterministicFiniteAutomaton()
    for label, box in rsm.boxes.items():
        for source, symbol, target in box.dfa:
            nfa.add_transition(State((label, source)), symbol, State((label, target)))
        # регистрируем и состояния без переходов, например старт бокса для S -> $
        for start_state in box.dfa.start_states:
            nfa.add_start_state(State((label, start_state)))
        for final_state in box.dfa.final_states:
            nfa.add_final_state(State((label, final_state)))

    rsm_fa = AdjacencyMatrixFA(nfa)

    def box_indices(label: Symbol, box_states: set[State]) -> set[int]:
        return {rsm_fa.state_to_index[State((label, state))] for state in box_states}

    box_bounds = {
        label: (
            box_indices(label, box.dfa.start_states),
            box_indices(label, box.dfa.final_states),
        )
        for label, box in rsm.boxes.items()
    }
    return rsm_fa, box_bounds


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """То же, что hellings_based_cfpq, но грамматика задана RSM.

    Пересекаем RSM с графом через тензорное произведение. Если в замыкании
    пересечения из (старт бокса N, u) достижимо (финал бокса N, v), то из N
    выводится слово на пути из u в v: добавляем в граф ребро u -N-> v
    и повторяем, пока граф растёт.
    """
    rsm_fa, box_bounds = _rsm_to_adjacency_matrix_fa(rsm)
    graph_fa = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    nodes_count = len(graph_fa.states)

    def node_block(rsm_index: int) -> slice:
        # intersect_automata кладёт пару (q, v) в индекс q * |V| + v
        return slice(rsm_index * nodes_count, (rsm_index + 1) * nodes_count)

    for label in box_bounds:
        graph_fa.matrices.setdefault(
            label, csr_array((nodes_count, nodes_count), dtype=bool)
        )

    changed = True
    while changed:
        changed = False
        closure = intersect_automata(rsm_fa, graph_fa).transitive_closure()
        for label, (box_starts, box_finals) in box_bounds.items():
            derived = graph_fa.matrices[label]
            for box_start in box_starts:
                for box_final in box_finals:
                    derived = (
                        derived + closure[node_block(box_start), node_block(box_final)]
                    )
            if derived.nnz != graph_fa.matrices[label].nnz:
                graph_fa.matrices[label] = derived
                changed = True

    initial_matrix = graph_fa.matrices[rsm.initial_label]
    return {
        (graph_fa.states[row].value, graph_fa.states[column].value)
        for row, column in zip(*initial_matrix.nonzero())
        if row in graph_fa.start_indices and column in graph_fa.final_indices
    }


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    """То же, что hellings_based_cfpq, но обход RSM по графу в духе GLL.

    Дескриптор (состояние RSM, вершина графа, узел GSS) - точка разбора.
    Узел GSS (нетерминал, вершина) - вызов бокса нетерминала из этой вершины,
    его рёбра ведут к местам возврата: (состояние RSM после вызова, вызвавший узел).
    """
    nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    graph_edges = defaultdict(lambda: defaultdict(set))
    for source, symbol, target in nfa:
        graph_edges[source.value][symbol.value].add(target.value)

    box_transitions = defaultdict(list)
    box_starts, all_box_finals = defaultdict(set), set()
    for label, box in rsm.boxes.items():
        for source, symbol, target in box.dfa:
            box_transitions[(label, source)].append((symbol.value, (label, target)))
        box_starts[label] = {(label, state) for state in box.dfa.start_states}
        all_box_finals |= {(label, state) for state in box.dfa.final_states}
    label_by_name = {label.value: label for label in rsm.boxes}

    return_points = defaultdict(set)
    # узел GSS -> вершины графа, в которых бокс уже завершился
    popped = defaultdict(set)
    visited = set()
    worklist = []

    def add_descriptor(rsm_state, node, gss_node) -> None:
        descriptor = (rsm_state, node, gss_node)
        if descriptor not in visited:
            visited.add(descriptor)
            worklist.append(descriptor)

    roots = {(rsm.initial_label, state.value) for state in nfa.start_states}
    for root in roots:
        _, start_node = root
        for rsm_state in box_starts[rsm.initial_label]:
            add_descriptor(rsm_state, start_node, root)

    while worklist:
        rsm_state, node, gss_node = worklist.pop()
        if rsm_state in all_box_finals and node not in popped[gss_node]:
            popped[gss_node].add(node)
            for return_state, caller in return_points[gss_node]:
                add_descriptor(return_state, node, caller)
        for symbol, next_state in box_transitions[rsm_state]:
            if symbol in label_by_name:
                callee_label = label_by_name[symbol]
                callee = (callee_label, node)
                return_point = (next_state, gss_node)
                if return_point not in return_points[callee]:
                    return_points[callee].add(return_point)
                    # вызов мог завершиться раньше, чем появилась эта точка возврата
                    for end_node in popped[callee]:
                        add_descriptor(next_state, end_node, gss_node)
                for start_state in box_starts[callee_label]:
                    add_descriptor(start_state, node, callee)
            else:
                for target in graph_edges[node][symbol]:
                    add_descriptor(next_state, target, gss_node)

    final_values = {state.value for state in nfa.final_states}
    return {
        (start, end)
        for _, start in roots
        for end in popped[(rsm.initial_label, start)]
        if end in final_values
    }
