#!/usr/bin/env python3
"""
Pytest configuration for on2vec tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_owl_file(temp_dir):
    """Create a mock OWL file for testing."""
    owl_content = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://example.org/test"/>

    <owl:Class rdf:about="http://example.org/ClassA">
        <rdfs:label>Class A</rdfs:label>
        <rdfs:comment>A test class</rdfs:comment>
    </owl:Class>

    <owl:Class rdf:about="http://example.org/ClassB">
        <rdfs:label>Class B</rdfs:label>
        <rdfs:comment>Another test class</rdfs:comment>
        <rdfs:subClassOf rdf:resource="http://example.org/ClassA"/>
    </owl:Class>

    <owl:Class rdf:about="http://example.org/ClassC">
        <rdfs:label>Class C</rdfs:label>
        <rdfs:subClassOf rdf:resource="http://example.org/ClassB"/>
    </owl:Class>
</rdf:RDF>"""

    owl_file = temp_dir / "test.owl"
    owl_file.write_text(owl_content)
    return str(owl_file)


@pytest.fixture
def mock_parquet_file(temp_dir):
    """Create a mock parquet file for testing."""
    parquet_file = temp_dir / "test_embeddings.parquet"
    # Just create an empty file - real tests should mock the parquet operations
    parquet_file.touch()
    return str(parquet_file)