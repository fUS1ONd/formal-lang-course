import networkx as nx
import pytest
from networkx.algorithms.isomorphism import categorical_multiedge_match

from project.graph_utils import (
    GraphInfo,
    get_graph_info,
    get_graph_info_by_name,
    save_two_cycles_graph,
)


def test_get_graph_info_counts_distinct_labels():
    graph = nx.MultiDiGraph()
    graph.add_edge(0, 1, label="a")
    graph.add_edge(0, 1, label="b")
    graph.add_edge(1, 2, label="a")

    assert get_graph_info(graph) == GraphInfo(3, 3, {"a", "b"})


def test_get_graph_info_empty_graph():
    assert get_graph_info(nx.MultiDiGraph()) == GraphInfo(0, 0, set())


def test_get_graph_info_by_name():
    assert get_graph_info_by_name("bzip") == GraphInfo(632, 556, {"a", "d"})


@pytest.mark.parametrize(
    "first_cycle_nodes_count, second_cycle_nodes_count, labels",
    [(1, 1, ("a", "b")), (3, 5, ("x", "y")), (42, 29, ("a", "b"))],
)
def test_save_two_cycles_graph(
    tmp_path, first_cycle_nodes_count, second_cycle_nodes_count, labels
):
    path = tmp_path / "two_cycles.dot"

    save_two_cycles_graph(
        first_cycle_nodes_count, second_cycle_nodes_count, labels, path
    )

    saved_graph = nx.nx_pydot.read_dot(path)
    expected_graph = build_two_cycles_graph(
        first_cycle_nodes_count, second_cycle_nodes_count, labels
    )
    assert nx.is_isomorphic(
        saved_graph,
        expected_graph,
        edge_match=categorical_multiedge_match("label", None),
    )


def build_two_cycles_graph(
    first_cycle_nodes_count: int,
    second_cycle_nodes_count: int,
    labels: tuple[str, str],
) -> nx.MultiDiGraph:
    # оба цикла проходят через общую вершину 0
    common_node = 0
    first_cycle = [common_node, *range(1, first_cycle_nodes_count + 1)]
    second_cycle = [
        common_node,
        *range(
            first_cycle_nodes_count + 1,
            first_cycle_nodes_count + second_cycle_nodes_count + 1,
        ),
    ]

    graph = nx.MultiDiGraph()
    nx.add_cycle(graph, first_cycle, label=labels[0])
    nx.add_cycle(graph, second_cycle, label=labels[1])
    return graph
