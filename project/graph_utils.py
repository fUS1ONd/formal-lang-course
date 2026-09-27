from dataclasses import dataclass
from pathlib import Path

import cfpq_data
import networkx as nx


@dataclass
class GraphInfo:
    nodes_count: int
    edges_count: int
    labels: set[str]


def get_graph_info_by_name(name: str) -> GraphInfo:
    path = cfpq_data.download(name)
    graph = cfpq_data.graph_from_csv(path)
    return get_graph_info(graph)


def get_graph_info(graph: nx.MultiDiGraph) -> GraphInfo:
    labels = {label for _, _, label in graph.edges(data="label")}
    return GraphInfo(graph.number_of_nodes(), graph.number_of_edges(), labels)


def save_two_cycles_graph(
    first_cycle_nodes_count: int,
    second_cycle_nodes_count: int,
    labels: tuple[str, str],
    path: Path | str,
) -> None:
    graph = cfpq_data.labeled_two_cycles_graph(
        first_cycle_nodes_count, second_cycle_nodes_count, labels=labels
    )
    nx.nx_pydot.to_pydot(graph).write_raw(path)
