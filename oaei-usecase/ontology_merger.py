"""
OWL ontology merger
"""

import os
import torch
from owlready2 import get_ontology, World
import logging
from typing import Dict, Tuple
import argparse
import rdflib

logger = logging.getLogger(__name__)

class OntologyMerger:
    """Handles merging of two OWL ontologies with conflict resolution"""
    
    def __init__(self, ontology1_path: str, ontology2_path: str):
        # Use single world for proper cross-ontology resolution
        self.world = World()
        
        self.onto1 = self._load_ontology(ontology1_path, self.world)
        self.onto2 = self._load_ontology(ontology2_path, self.world)
        
        # Initialize merged ontology in the same world
        self.merged_onto = self.world.get_ontology("http://merged.ontology/")
        

    def _load_ontology(self, path: str, world: World):
        """Load ontology into specified world with error handling"""
        try:
            ontology_iri = f"file://{os.path.abspath(path)}"
            print(f"Loading ontology from {ontology_iri}")
            return world.get_ontology(ontology_iri).load()
        except Exception as e:
            logger.error(f"Failed to load ontology {path}: {e}")
            raise


    def merge_axioms(self):
        """Merge all RDF triples (axioms) from both ontologies into merged ontology"""
        # Add triples inside a 'with' block for the merged ontology
        with self.merged_onto:
            for onto in [self.onto1, self.onto2]:
                for triple in onto.world.as_rdflib_graph().triples((None, None, None)):
                    self.merged_onto.world.as_rdflib_graph().add(triple)


    def save_merged(self, output_path: str):
        """Save merged ontology to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.merged_onto.save(
            file=output_path,
            format="rdfxml"
        )

def merge_ontologies(
    ontology1_path: str,
    ontology2_path: str,
    output_path: str
) -> Tuple[bool, str]:
    """Merge two OWL ontologies into a new combined ontology
    
    Args:
        ontology1_path: Path to first ontology file
        ontology2_path: Path to second ontology file
        output_path: Path to save merged ontology
        
    Returns:
        Tuple: (success boolean, message/output path)
    """
    try:
        merger = OntologyMerger(ontology1_path, ontology2_path)
        merger.merge_axioms()
        merger.save_merged(output_path)
        return True, output_path
    except Exception as e:
        logger.error(f"Merge failed: {str(e)}", exc_info=True)
        return False, f"{str(e)}\n\nFull traceback in logs"

# Original ontology processing functions below...
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Merge two OWL ontologies.")
    parser.add_argument("ontology1", help="Path to first ontology file")
    parser.add_argument("ontology2", help="Path to second ontology file")
    parser.add_argument("output", help="Path to save merged ontology")
    args = parser.parse_args()

    success, message = merge_ontologies(args.ontology1, args.ontology2, args.output)
    if success:
        print(f"Merged ontology saved to: {message}")
    else:
        print(f"Ontology merge failed: {message}")