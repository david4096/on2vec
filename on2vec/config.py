#!/usr/bin/env python3
"""
Configuration management for on2vec CLI.

Supports YAML and JSON configuration files for default parameters.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            if config_file.suffix.lower() in ['.yml', '.yaml']:
                if yaml is None:
                    raise ValueError("PyYAML not installed. Install with: pip install PyYAML")
                config = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                config = json.load(f)
            else:
                # Try to auto-detect format
                content = f.read()
                f.seek(0)
                try:
                    if yaml is not None:
                        config = yaml.safe_load(f)
                    else:
                        raise Exception("YAML not available")
                except Exception:
                    f.seek(0)
                    config = json.load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config or {}

    except Exception as e:
        raise ValueError(f"Failed to parse configuration file {config_path}: {e}")


def find_default_config() -> Optional[str]:
    """
    Find default configuration file in common locations.

    Returns:
        Path to config file if found, None otherwise
    """
    search_paths = [
        Path.cwd() / "on2vec.yml",
        Path.cwd() / "on2vec.yaml",
        Path.cwd() / "on2vec.json",
        Path.cwd() / ".on2vec.yml",
        Path.cwd() / ".on2vec.yaml",
        Path.cwd() / ".on2vec.json",
        Path.home() / ".config" / "on2vec" / "config.yml",
        Path.home() / ".config" / "on2vec" / "config.yaml",
        Path.home() / ".config" / "on2vec" / "config.json",
        Path.home() / ".on2vec.yml",
        Path.home() / ".on2vec.yaml",
        Path.home() / ".on2vec.json",
    ]

    for config_path in search_paths:
        if config_path.exists():
            return str(config_path)

    return None


def merge_config_with_args(config: Dict[str, Any], args: Any, command: str) -> Any:
    """
    Merge configuration file values with command line arguments.
    Command line arguments take precedence.

    Args:
        config: Configuration dictionary
        args: Parsed command line arguments
        command: Command name

    Returns:
        Updated arguments with config defaults
    """
    if not config:
        return args

    # Get command-specific config
    command_config = config.get(command, {})
    global_config = config.get('global', {})

    # Apply global config first, then command-specific
    for config_dict in [global_config, command_config]:
        for key, value in config_dict.items():
            # Convert hyphenated keys to underscored (CLI convention)
            key = key.replace('-', '_')

            # Only set if not already set by command line
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)
            elif hasattr(args, key) and key in ['epochs', 'hidden_dim', 'out_dim'] and getattr(args, key) == getattr(type(args), key, None):
                # Handle default values that might not be None
                setattr(args, key, value)

    return args


def create_sample_config() -> str:
    """
    Create a sample configuration file with common settings.

    Returns:
        Sample configuration as YAML string
    """
    sample_config = {
        'global': {
            'verbose': False,
            'quiet': False
        },
        'train': {
            'model-type': 'gcn',
            'hidden-dim': 128,
            'out-dim': 64,
            'epochs': 100,
            'loss-fn': 'triplet',
            'text-model': 'all-MiniLM-L6-v2'
        },
        'hf': {
            'output-dir': './hf_models',
            'fusion': 'concat',
            'base-model': 'all-MiniLM-L6-v2',
            'epochs': 100,
            'model-type': 'gcn',
            'hidden-dim': 128,
            'out-dim': 64,
            'license': 'apache-2.0'
        },
        'benchmark': {
            'output-dir': './mteb_results',
            'batch-size': 32,
            'quick': False
        },
        'visualize': {
            'neighbors': 15,
            'min-dist': 0.1
        }
    }

    if yaml is not None:
        return yaml.dump(sample_config, default_flow_style=False, sort_keys=True)
    else:
        return json.dumps(sample_config, indent=2, sort_keys=True)


def save_sample_config(output_path: str = "on2vec.yml"):
    """
    Save a sample configuration file.

    Args:
        output_path: Where to save the config file
    """
    sample = create_sample_config()

    # Use JSON if YAML not available and user didn't specify extension
    if yaml is None and not any(output_path.endswith(ext) for ext in ['.yml', '.yaml', '.json']):
        output_path = output_path.replace('.yml', '.json')

    with open(output_path, 'w') as f:
        f.write("# on2vec Configuration File\n")
        f.write("# This file sets default values for on2vec commands\n")
        f.write("# Command line arguments override these settings\n\n")
        f.write(sample)

    print(f"📝 Sample configuration saved to {output_path}")
    print(f"💡 Edit this file to customize your defaults")


if __name__ == "__main__":
    # Allow running as script to generate sample config
    import sys

    if len(sys.argv) > 1:
        save_sample_config(sys.argv[1])
    else:
        save_sample_config()