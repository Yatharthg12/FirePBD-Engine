def test_shortest_path_across_connected_zones(simple_topology):
    _, graph, _ = simple_topology
    assert graph.shortest_path("Z0001", "Z0003") == ["Z0001", "Z0002", "Z0003"]


def test_shortest_path_to_any_exit(simple_topology):
    model, graph, _ = simple_topology
    assert graph.shortest_path_to_any_exit("Z0001", model.exit_zone_ids) == [
        "Z0001",
        "Z0002",
        "Z0003",
    ]


def test_graph_bottleneck_metrics(simple_topology):
    _, graph, _ = simple_topology
    centrality = graph.compute_betweenness_centrality()
    bottlenecks = graph.identify_bottleneck_edges()

    assert centrality["Z0002"] > centrality["Z0001"]
    assert bottlenecks
    assert bottlenecks[0][:2] in {("Z0001", "Z0002"), ("Z0002", "Z0003")}
