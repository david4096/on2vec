#!/usr/bin/env python3
"""
Tests for on2vec core workflows.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from on2vec.workflows import (
    train_and_embed_workflow,
    embed_with_trained_model,
    train_model_only
)


class TestWorkflows:
    """Test suite for core workflow functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Mock OWL file
        self.owl_file = str(self.temp_path / "test.owl")
        with open(self.owl_file, 'w') as f:
            f.write("""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
    <owl:Ontology rdf:about="http://example.org/test"/>
    <owl:Class rdf:about="http://example.org/ClassA">
        <rdfs:label>Class A</rdfs:label>
    </owl:Class>
    <owl:Class rdf:about="http://example.org/ClassB">
        <rdfs:label>Class B</rdfs:label>
        <rdfs:subClassOf rdf:resource="http://example.org/ClassA"/>
    </owl:Class>
</rdf:RDF>""")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('on2vec.workflows.train_ontology_embeddings')
    @patch('on2vec.workflows.embed_same_ontology')
    def test_train_and_embed_workflow_standard(self, mock_embed, mock_train):
        """Test standard training workflow without text features."""
        # Mock return values
        mock_train.return_value = {
            'model_path': 'test_model.pt',
            'structural_dim': 64
        }
        mock_embed.return_value = {
            'node_ids': ['class1', 'class2'],
            'embeddings': [[1, 2], [3, 4]]
        }

        result = train_and_embed_workflow(
            owl_file=self.owl_file,
            model_type='gcn',
            hidden_dim=16,
            out_dim=8,
            epochs=10,
            output='test_embeddings.parquet',
            model_output='test_model.pt'
        )

        # Verify training was called with correct parameters
        mock_train.assert_called_once()
        mock_embed.assert_called_once_with(
            model_path='test_model.pt',
            owl_file=self.owl_file,
            output_file='test_embeddings.parquet'
        )

        # Check result structure
        assert 'training_result' in result
        assert 'embedding_result' in result
        assert 'model_path' in result
        assert 'embeddings_path' in result
        assert result['num_embeddings'] == 2

    @patch('on2vec.workflows.train_text_augmented_ontology_embeddings')
    @patch('on2vec.workflows.embed_same_ontology')
    def test_train_and_embed_workflow_with_text(self, mock_embed, mock_train_text):
        """Test training workflow with text features enabled."""
        mock_train_text.return_value = {
            'model_path': 'test_model.pt',
            'structural_dim': 64,
            'text_dim': 384,
            'text_features_extracted': 5
        }
        mock_embed.return_value = {
            'node_ids': ['class1', 'class2'],
            'embeddings': [[1, 2], [3, 4]]
        }

        result = train_and_embed_workflow(
            owl_file=self.owl_file,
            use_text_features=True,
            text_model_name='all-MiniLM-L6-v2',
            fusion_method='concat'
        )

        mock_train_text.assert_called_once()
        assert result['num_embeddings'] == 2

    @patch('on2vec.workflows.embed_ontology_with_model')
    def test_embed_with_trained_model(self, mock_embed):
        """Test embedding generation with pre-trained model."""
        mock_embed.return_value = {
            'node_ids': ['class1', 'class2', 'class3'],
            'embeddings': [[1, 2], [3, 4], [5, 6]]
        }

        result = embed_with_trained_model(
            model_path='test_model.pt',
            owl_file=self.owl_file,
            output_file='test_output.parquet'
        )

        mock_embed.assert_called_once_with(
            model_path='test_model.pt',
            owl_file=self.owl_file,
            output_file='test_output.parquet'
        )

        assert result['num_embeddings'] == 3
        assert result['output_file'] == 'test_output.parquet'

    @patch('on2vec.workflows.train_ontology_embeddings')
    def test_train_model_only(self, mock_train):
        """Test training only without embedding generation."""
        mock_train.return_value = {
            'model_path': 'test_model.pt',
            'structural_dim': 128
        }

        result = train_model_only(
            owl_file=self.owl_file,
            model_output='test_model.pt',
            model_type='gat',
            hidden_dim=64,
            out_dim=32
        )

        mock_train.assert_called_once()
        assert result['model_path'] == 'test_model.pt'

    def test_workflow_parameter_validation(self):
        """Test parameter validation in workflows."""
        # Test with invalid model type (should work due to parameter passing)
        with pytest.raises((ValueError, FileNotFoundError, Exception)):
            train_model_only(
                owl_file="nonexistent.owl",
                model_output="test.pt"
            )