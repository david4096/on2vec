#!/usr/bin/env python3
"""
Tests for on2vec CLI functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

from on2vec.cli import (
    create_parser,
    main,
    run_train_command,
    run_embed_command,
    run_hf_command,
    run_hf_train_command,
    run_hf_create_command,
    run_hf_test_command
)


class TestCLI:
    """Test suite for CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_create_parser(self):
        """Test parser creation."""
        parser = create_parser()
        assert parser.prog == 'on2vec'
        assert 'Generate vector embeddings' in parser.description

    def test_parser_help(self, capsys):
        """Test parser help output."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['--help'])
        captured = capsys.readouterr()
        assert 'on2vec' in captured.out

    def test_train_parser(self):
        """Test train command parser."""
        parser = create_parser()
        args = parser.parse_args([
            'train', 'test.owl', '--output', 'model.pt',
            '--model-type', 'gcn', '--epochs', '50'
        ])

        assert args.command == 'train'
        assert args.ontology == 'test.owl'
        assert args.output == 'model.pt'
        assert args.model_type == 'gcn'
        assert args.epochs == 50

    def test_embed_parser(self):
        """Test embed command parser."""
        parser = create_parser()
        args = parser.parse_args([
            'embed', 'model.pt', 'ontology.owl', '--output', 'embeddings.parquet'
        ])

        assert args.command == 'embed'
        assert args.model == 'model.pt'
        assert args.ontology == 'ontology.owl'
        assert args.output == 'embeddings.parquet'

    def test_hf_parser(self):
        """Test HuggingFace end-to-end command parser."""
        parser = create_parser()
        args = parser.parse_args([
            'hf', 'ontology.owl', 'my-model',
            '--base-model', 'all-MiniLM-L6-v2',
            '--epochs', '100'
        ])

        assert args.command == 'hf'
        assert args.ontology == 'ontology.owl'
        assert args.model_name == 'my-model'
        assert args.base_model == 'all-MiniLM-L6-v2'
        assert args.epochs == 100

    @patch('on2vec.cli.run_train_command')
    def test_main_train_command(self, mock_run_train):
        """Test main function routing to train command."""
        mock_run_train.return_value = 0

        result = main(['train', 'test.owl', '--output', 'model.pt'])

        mock_run_train.assert_called_once()
        assert result == 0

    @patch('on2vec.workflows.train_model_only')
    def test_run_train_command_success(self, mock_train):
        """Test successful train command execution."""
        mock_train.return_value = {'model_path': 'test_model.pt'}

        # Create mock args
        args = MagicMock()
        args.ontology = 'test.owl'
        args.output = 'model.pt'
        args.model_type = 'gcn'
        args.hidden_dim = 128
        args.out_dim = 64
        args.epochs = 100
        args.loss_fn = 'triplet'
        args.use_multi_relation = False
        args.use_text_features = False
        args.text_model = 'all-MiniLM-L6-v2'

        result = run_train_command(args)

        mock_train.assert_called_once()
        assert result == 0

    @patch('on2vec.workflows.train_model_only')
    def test_run_train_command_failure(self, mock_train):
        """Test train command execution failure."""
        mock_train.side_effect = Exception("Training failed")

        # Create mock args
        args = MagicMock()
        args.ontology = 'test.owl'
        args.output = 'model.pt'
        args.model_type = 'gcn'
        args.hidden_dim = 128
        args.out_dim = 64
        args.epochs = 100
        args.loss_fn = 'triplet'
        args.use_multi_relation = False
        args.use_text_features = False
        args.text_model = 'all-MiniLM-L6-v2'

        result = run_train_command(args)

        assert result == 1

    @patch('on2vec.workflows.embed_with_trained_model')
    def test_run_embed_command_success(self, mock_embed):
        """Test successful embed command execution."""
        mock_embed.return_value = {
            'output_file': 'embeddings.parquet',
            'num_embeddings': 100
        }

        # Create mock args
        args = MagicMock()
        args.model = 'model.pt'
        args.ontology = 'test.owl'
        args.output = 'embeddings.parquet'

        result = run_embed_command(args)

        mock_embed.assert_called_once_with(
            model_path='model.pt',
            owl_file='test.owl',
            output_file='embeddings.parquet'
        )
        assert result == 0

    @patch('on2vec.huggingface_workflows.end_to_end_workflow')
    def test_run_hf_command_success(self, mock_workflow):
        """Test successful HuggingFace end-to-end command."""
        mock_workflow.return_value = True

        # Create mock args
        args = MagicMock()
        args.ontology = 'test.owl'
        args.model_name = 'test-model'
        args.output_dir = './hf_models'
        args.base_model = 'all-MiniLM-L6-v2'
        args.fusion = 'concat'
        args.epochs = 100
        args.skip_training = False
        args.skip_testing = False

        result = run_hf_command(args)

        mock_workflow.assert_called_once()
        assert result == 0

    @patch('on2vec.huggingface_workflows.train_ontology_with_text')
    def test_run_hf_train_command_success(self, mock_train):
        """Test successful HuggingFace train command."""
        mock_train.return_value = True

        # Create mock args
        args = MagicMock()
        args.ontology = 'test.owl'
        args.output = 'embeddings.parquet'
        args.text_model = 'all-MiniLM-L6-v2'
        args.epochs = 100
        args.model_type = 'gcn'
        args.hidden_dim = 128
        args.out_dim = 64

        result = run_hf_train_command(args)

        mock_train.assert_called_once()
        assert result == 0

    @patch('on2vec.huggingface_workflows.create_hf_model')
    def test_run_hf_create_command_success(self, mock_create):
        """Test successful HuggingFace create command."""
        mock_create.return_value = './hf_models/test-model'

        # Create mock args
        args = MagicMock()
        args.embeddings = 'embeddings.parquet'
        args.model_name = 'test-model'
        args.output_dir = './hf_models'
        args.base_model = 'all-MiniLM-L6-v2'
        args.fusion = 'concat'
        args.ontology = 'test.owl'

        result = run_hf_create_command(args)

        mock_create.assert_called_once()
        assert result == 0

    @patch('on2vec.huggingface_workflows.validate_hf_model')
    def test_run_hf_test_command_success(self, mock_test):
        """Test successful HuggingFace test command."""
        mock_test.return_value = True

        # Create mock args
        args = MagicMock()
        args.model_path = './hf_models/test-model'
        args.queries = ['test query 1', 'test query 2']

        result = run_hf_test_command(args)

        mock_test.assert_called_once()
        assert result == 0

    def test_main_no_args(self, capsys):
        """Test main function with no arguments shows help."""
        result = main([])

        # Should return 0 and print help
        assert result == 0

    def test_main_help(self, capsys):
        """Test main function with --help."""
        result = main(['--help'])

        # Should return 0 and print help
        assert result == 0

    @patch('on2vec.cli.run_train_command')
    def test_main_command_exception(self, mock_run_train):
        """Test main function handling command exceptions."""
        mock_run_train.side_effect = Exception("Command failed")

        result = main(['train', 'test.owl', '--output', 'model.pt'])

        assert result == 1

    def test_main_unknown_command(self, capsys):
        """Test main function with unknown command."""
        with pytest.raises(SystemExit) as exc_info:
            main(['unknown-command'])

        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "invalid choice" in captured.err

    def test_version_argument(self):
        """Test --version argument."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['--version'])