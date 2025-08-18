import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from matplotlib.lines import Line2D
from collections import defaultdict
import numpy as np

# flow contents to (edge color, text color) mapping
color_map = {
    "Electricity": ("yellow", "black"),
    "UntreatedSewage": ("saddlebrown", "white"),
    "PrimaryEffluent": ("saddlebrown", "white"),
    "SecondaryEffluent": ("saddlebrown", "white"),
    "TertiaryEffluent": ("saddlebrown", "white"),
    "TreatedSewage": ("green", "black"),
    "WasteActivatedSludge": ("black", "white"),
    "PrimarySludge": ("black", "white"),
    "TWAS": ("black", "white"),
    "TPS": ("black", "white"),
    "Scum": ("black", "white"),
    "SludgeBlend": ("black", "white"),
    "ThickenedSludgeBlend": ("black", "white"),
    "Biogas": ("red", "black"),
    "GasBlend": ("red", "black"),
    "NaturalGas": ("gray", "black"),
    "Seawater": ("aqua", "black"),
    "Brine": ("aqua", "black"),
    "SurfaceWater": ("cornflowerblue", "black"),
    "Groundwater": ("cornflowerblue", "black"),
    "Stormwater": ("cornflowerblue", "black"),
    "NonpotableReuse": ("purple", "black"),
    "DrinkingWater": ("blue", "white"),
    "PotableReuse": ("blue", "white"),
    "FatOilGrease": ("orange", "black"),
    "FoodWaste": ("orange", "black"),
}


def assign_layers(g):
    # Initialize all node layers to None
    layers = {node: None for node in g.nodes()}
    # Find all source nodes (no incoming edges)
    sources = [n for n in g.nodes() if g.in_degree(n) == 0]
    # Assign layer 0 to sources
    for s in sources:
        layers[s] = 0
    
    # BFS to assign layers - FIXED: track processed nodes to prevent infinite loops
    queue = sources[:]
    processed_nodes = set(sources)  # Track which nodes we've already processed
    processed_count = 0
    
    while queue:
        current = queue.pop(0)
        processed_count += 1

        
        for succ in g.successors(current):
            # Only process successors that haven't been processed yet
            if succ not in processed_nodes:
                if layers[succ] is None or layers[succ] < layers[current] + 1:
                    layers[succ] = layers[current] + 1
                    queue.append(succ)
                    processed_nodes.add(succ)  # Mark as processed
    
    # Count nodes in each layer
    layer_counts = {}
    for layer in layers.values():
        if layer is not None:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
    
    return layers


def local_draw_graph(network, pyvis=False, output_file=None, layout="spring", spread=1.0):
    """Draw all of the nodes and connections in the given network

    Parameters
    ----------
    network : Network
        `Network` object to draw

    pyvis : bool
        Whether to draw the graph with PyVis or Networkx.
        False (networkx) by default

    output_file : str
        Path to the desired output.
        Default is None, meaning the file will be saved as `networkd.id` + extension

    layout : str
        Layout algorithm to use for NetworkX drawing.
        "spring" by default

    spread : float
        Spread parameter for NetworkX layout.
        1.0 by default
    """
    # create empty graph
    g = nx.MultiDiGraph()

    # add list of nodes and edges to graph
    g.add_nodes_from(network.nodes.__iter__())

    flow_colors = defaultdict(str)
    font_colors = defaultdict(str)
    
    edge_count = 0
    for id, connection in network.connections.items():
        try:
            flow_color = color_map[connection.contents.name][0]
            font_color = color_map[connection.contents.name][1]
        except KeyError:
            flow_color = "black"
            font_color = "white"

        flow_colors[connection.contents.name] = flow_color
        font_colors[connection.contents.name] = font_color

        g.add_edge(
            connection.source.id, connection.destination.id, color=flow_color, label=id
        )
        edge_count += 1

        if connection.bidirectional:
            g.add_edge(
                connection.destination.id,
                connection.source.id,
                color=flow_color,
                label=id,
            )
            edge_count += 1
    

    colors = list(flow_colors.values())
    labels = list(flow_colors.keys())
    if pyvis:
        nt = Network("500px", "500px", directed=True, notebook=False)

        # create legend based on https://github.com/WestHealth/pyvis/issues/50
        num_legend_nodes = len(flow_colors)
        num_actual_nodes = len(g.nodes())
        step = 50
        x = -300
        y = -250
        legend_nodes = [
            (
                num_actual_nodes + legend_node,
                {
                    "color": colors[legend_node],
                    "label": labels[legend_node],
                    "size": 30,
                    "physics": False,
                    "x": x,
                    "y": f"{y + legend_node*step}px",
                    "shape": "box",
                    "font": {"size": 12, "color": font_colors[legend_node]},
                },
            )
            for legend_node in range(num_legend_nodes)
        ]
        g.add_nodes_from(legend_nodes)

        nt.from_nx(g)
        if output_file:
            nt.show(output_file, notebook=False)
        else:
            nt.show(network.id + ".html", notebook=False)
    else:
        # create legend
        custom_lines = []
        for color in colors:
            custom_lines.append(Line2D([0], [0], color=color, lw=4))
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.legend(custom_lines, labels)

        edge_colors = []
        edges = g.edges()
        node_to_node = [g[u][v] for u, v in edges]
        for edge_dict in node_to_node:
            for _, edge in edge_dict.items():
                edge_colors.append(edge["color"])

        # Use layout and spread
        if layout == "multipartite":
            layer_mapping = assign_layers(g)
            nx.set_node_attributes(g, layer_mapping, "layer")
            pos = nx.multipartite_layout(g, subset_key="layer", align="vertical")
        elif layout == "spring":
            pos = nx.spring_layout(g, k=spread)
        elif layout == "shell":
            pos = nx.shell_layout(g)
        elif layout == "circular":
            pos = nx.circular_layout(g)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(g)
        elif layout == "horizontal_hierarchy":
            pos = nx.spring_layout(g, k=spread, seed=42)
        else:
            pos = nx.spring_layout(g, k=spread)

        nx.draw(g, pos, with_labels=True, edge_color=edge_colors)

        plt.axis("off")
        axis = plt.gca()
        axis.set_xlim([1.2 * x for x in axis.get_xlim()])
        axis.set_ylim([1.2 * y for y in axis.get_ylim()])
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file)
        else:
            plt.savefig(network.id + ".png")