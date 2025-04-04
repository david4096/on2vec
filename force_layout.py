import argparse
import networkx as nx
import matplotlib.pyplot as plt
from owlready2 import get_ontology
import logging
import os
import pandas as pd
from networkx.drawing.nx_pydot import graphviz_layout

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_graph_from_owl(owl_file):
    logger.info(f"Loading OWL ontology from {owl_file}")
    
    # Parse the OWL ontology
    try:
        onto = get_ontology(owl_file).load()
        logger.info(f"Successfully loaded ontology: {owl_file}")
    except Exception as e:
        logger.error(f"Failed to load ontology: {owl_file}. Error: {e}")
        raise

    # Create a directed graph using NetworkX
    G = nx.DiGraph()

    # Add classes as nodes
    logger.info("Adding classes as nodes to the graph")
    for cls in onto.classes():
        G.add_node(cls.name)

    # Add subclass relationships as edges
    logger.info("Adding subclass relationships as edges to the graph")
    for cls in onto.classes():
        for subclass in cls.subclasses():
            G.add_edge(cls.name, subclass.name)

    logger.info(f"Graph construction complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def graphviz_layout_ontology(G, owl_file, output_image, output_parquet):
    logger.info("Computing Graphviz layout (twopi) for the graph")
    
    # Compute the Graphviz layout (dot layout)
    try:
        pos = graphviz_layout(G, prog="dot")
        logger.info("Layout computation successful")
    except Exception as e:
        logger.error(f"Failed to compute layout using Graphviz. Error: {e}")
        raise

    # Save node coordinates to Parquet
    if output_parquet:
        logger.info(f"Saving node coordinates to Parquet file: {output_parquet}")
        try:
            # Create a DataFrame with node IDs and positions
            df = pd.DataFrame({
                'node_id': list(pos.keys()),
                'x': [p[0] for p in pos.values()],
                'y': [p[1] for p in pos.values()]
            })
            df.to_parquet(output_parquet)
            logger.info(f"Node coordinates successfully saved to {output_parquet}")
        except Exception as e:
            logger.error(f"Failed to save node coordinates to Parquet. Error: {e}")
            raise

    # Draw the graph
    plt.figure(figsize=(16, 12))
    logger.info("Drawing the graph")
    try:
        nx.draw(G, pos, with_labels=True, labels={node: node for node in G.nodes()}, 
                node_color='lightblue', edge_color=[.3, .3, .5, .1], node_size=300, font_size=8)
        logger.info("Graph drawing complete")
    except Exception as e:
        logger.error(f"Failed to draw the graph. Error: {e}")
        raise

    # Add the ontology filename as the title
    ontology_name = os.path.basename(owl_file)
    plt.title(f"Graphviz Layout (twopi) for Ontology: {ontology_name}", fontsize=14)

    # Save or display the figure
    if output_image:
        try:
            plt.savefig(output_image)
            logger.info(f"Graph successfully saved to {output_image}")
        except Exception as e:
            logger.error(f"Failed to save graph to {output_image}. Error: {e}")
            raise
    else:
        logger.info("Displaying graph interactively")
        plt.show()

def main():
    # Argument parser for input file and optional output image/Parquet files
    parser = argparse.ArgumentParser(description="Graphviz (twopi) layout of an OWL ontology")
    parser.add_argument('owl_file', help="Path to the OWL file")
    parser.add_argument('--output_image', help="Path to save the output image (optional)", default=None)
    parser.add_argument('--output_parquet', help="Path to save the node coordinates as a Parquet file (optional)", default=None)
    args = parser.parse_args()

    # Build the graph from the OWL file
    G = build_graph_from_owl(args.owl_file)

    # Generate and visualize the Graphviz layout, and save coordinates to Parquet
    graphviz_layout_ontology(G, args.owl_file, args.output_image, args.output_parquet)

if __name__ == "__main__":
    main()
