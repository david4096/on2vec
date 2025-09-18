#!/usr/bin/env python3
"""
Tests for on2vec HuggingFace integration workflows.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from on2vec.huggingface_workflows import (
    train_ontology_with_text,
    create_hf_model,
    validate_embeddings,
    validate_hf_model,
    end_to_end_workflow
)


class TestHuggingFaceWorkflows:
    """Test suite for HuggingFace workflow functions."""

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
</rdf:RDF>""")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('on2vec.huggingface_workflows.train_and_embed_workflow')
    def test_train_ontology_with_text(self, mock_workflow):
        """Test text-augmented ontology training."""
        mock_workflow.return_value = {
            'training_result': {'model_path': 'test_model.pt'},
            'embedding_result': {'node_ids': ['class1', 'class2']},
            'model_path': 'test_model.pt',
            'embeddings_path': 'test_embeddings.parquet'
        }

        result = train_ontology_with_text(
            owl_file=self.owl_file,
            output_file="test_embeddings.parquet",
            text_model="all-MiniLM-L6-v2",
            epochs=10
        )

        mock_workflow.assert_called_once_with(
            owl_file=self.owl_file,
            model_type="gcn",
            hidden_dim=128,
            out_dim=64,
            epochs=10,
            output="test_embeddings.parquet",
            model_output="test_embeddings_model.pt",
            loss_fn="triplet",
            use_text_features=True,
            text_model_name="all-MiniLM-L6-v2",
            fusion_method="concat"
        )

        assert result is True

    @patch('on2vec.huggingface_workflows.inspect_parquet_metadata')
    @patch('on2vec.huggingface_workflows.get_embedding_info')
    def test_validate_embeddings_success(self, mock_get_info, mock_inspect):
        """Test successful embedding validation."""
        mock_get_info.return_value = {
            'num_embeddings': 1000,
            'text_dim': 384,
            'structural_dim': 64
        }

        # Create a mock parquet file
        embeddings_file = str(self.temp_path / "test_embeddings.parquet")
        Path(embeddings_file).touch()

        result = validate_embeddings(embeddings_file)

        mock_inspect.assert_called_once_with(embeddings_file)
        mock_get_info.assert_called_once_with(embeddings_file)

        assert result['num_embeddings'] == 1000
        assert result['text_dim'] == 384
        assert result['structural_dim'] == 64

    @patch('on2vec.huggingface_workflows.inspect_parquet_metadata')
    def test_validate_embeddings_failure(self, mock_inspect):
        """Test embedding validation failure."""
        mock_inspect.side_effect = Exception("Invalid parquet file")

        with pytest.raises(Exception):
            validate_embeddings("nonexistent_file.parquet")

    @patch('on2vec.huggingface_workflows.create_and_save_hf_model')
    @patch('on2vec.huggingface_workflows.create_model_card')
    @patch('on2vec.huggingface_workflows.create_upload_instructions')
    @patch('on2vec.huggingface_workflows.get_base_model_from_embeddings')
    @patch('on2vec.huggingface_workflows.validate_embeddings')
    def test_create_hf_model_success(self, mock_validate, mock_get_base,
                                   mock_upload_instr, mock_model_card, mock_create):
        """Test successful HuggingFace model creation."""
        # Setup mocks
        embeddings_file = str(self.temp_path / "test_embeddings.parquet")
        Path(embeddings_file).touch()

        mock_validate.return_value = {'num_embeddings': 100}
        mock_get_base.return_value = "all-MiniLM-L6-v2"
        mock_create.return_value = str(self.temp_path / "hf_model")

        result = create_hf_model(
            embeddings_file=embeddings_file,
            model_name="test-model",
            output_dir=str(self.temp_path),
            fusion_method="concat"
        )

        mock_validate.assert_called_once_with(embeddings_file)
        mock_create.assert_called_once()
        mock_model_card.assert_called_once()
        mock_upload_instr.assert_called_once()

        assert str(self.temp_path) in result

    @patch('sentence_transformers.SentenceTransformer')
    def test_test_model_success(self, mock_st):
        """Test successful model testing."""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 448
        import numpy as np
        mock_model.encode.return_value = np.array([[1, 2, 3], [4, 5, 6]])
        mock_st.return_value = mock_model

        # Mock cos_sim
        with patch('sentence_transformers.util.cos_sim') as mock_cos_sim:
            mock_similarity_tensor = MagicMock()
            mock_similarity_tensor.__getitem__.return_value.__getitem__.return_value.item.return_value = 0.8
            mock_cos_sim.return_value = mock_similarity_tensor

            model_path = str(self.temp_path / "test_model")
            Path(model_path).mkdir()

            result = validate_hf_model(model_path, ["test query 1", "test query 2"])

            assert result is True
            mock_st.assert_called_once_with(model_path)
            mock_model.encode.assert_called_once()

    @patch('on2vec.huggingface_workflows.train_ontology_with_text')
    @patch('on2vec.huggingface_workflows.validate_embeddings')
    @patch('on2vec.huggingface_workflows.create_hf_model')
    @patch('on2vec.huggingface_workflows.validate_hf_model')
    @patch('on2vec.huggingface_workflows.show_upload_instructions')
    def test_end_to_end_workflow_success(self, mock_show_upload, mock_test,
                                        mock_create, mock_validate, mock_train):
        """Test successful end-to-end workflow."""
        # Setup mocks
        mock_train.return_value = True
        mock_validate.return_value = {
            'num_embeddings': 500,
            'text_dim': 384,
            'structural_dim': 64
        }
        mock_create.return_value = str(self.temp_path / "hf_model")
        mock_test.return_value = True

        result = end_to_end_workflow(
            owl_file=self.owl_file,
            model_name="test-model",
            output_dir=str(self.temp_path),
            epochs=5,
            skip_testing=False
        )

        assert result is True
        mock_train.assert_called_once()
        mock_validate.assert_called_once()
        mock_create.assert_called_once()
        mock_test.assert_called_once()
        mock_show_upload.assert_called_once()

    @patch('on2vec.huggingface_workflows.validate_embeddings')
    def test_end_to_end_workflow_skip_training(self, mock_validate):
        """Test end-to-end workflow with training skipped."""
        mock_validate.return_value = {
            'num_embeddings': 500,
            'text_dim': 384,
            'structural_dim': 64
        }

        # Create mock embeddings file
        embeddings_file = str(self.temp_path / "existing_embeddings.parquet")
        Path(embeddings_file).touch()

        with patch('on2vec.huggingface_workflows.create_hf_model') as mock_create:
            mock_create.return_value = str(self.temp_path / "hf_model")

            with patch('on2vec.huggingface_workflows.show_upload_instructions'):
                result = end_to_end_workflow(
                    owl_file=self.owl_file,
                    model_name="test-model",
                    output_dir=str(self.temp_path),
                    embeddings_file=embeddings_file,
                    skip_training=True,
                    skip_testing=True
                )

                assert result is True
                mock_validate.assert_called_once_with(embeddings_file)