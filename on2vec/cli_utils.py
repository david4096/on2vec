#!/usr/bin/env python3
"""
Utility functions for CLI operations including validation, progress tracking, and formatting.
"""

import time
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
import logging

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Simple progress tracker for CLI operations."""

    def __init__(self, description: str = "Processing"):
        self.description = description
        self.start_time = None
        self.step_count = 0

    def start(self):
        """Start the progress tracker."""
        self.start_time = time.time()
        print(f"⏳ {self.description}...")

    def step(self, message: str = None):
        """Log a progress step."""
        self.step_count += 1
        if message:
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"   {self.step_count}. {message} ({elapsed:.1f}s)")

    def finish(self, success: bool = True, message: str = None):
        """Finish the progress tracker."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            status = "✅" if success else "❌"
            default_msg = "Completed" if success else "Failed"
            final_message = message or default_msg
            print(f"{status} {final_message} ({elapsed:.1f}s total)")


def validate_file_exists(filepath: str, file_type: str = "file") -> bool:
    """
    Validate that a file exists and provide helpful error messages.

    Args:
        filepath: Path to the file
        file_type: Type description for error message (e.g., "OWL file", "model file")

    Returns:
        True if file exists, False otherwise
    """
    path = Path(filepath)

    if not path.exists():
        print(f"❌ {file_type.capitalize()} not found: {filepath}")

        # Check if parent directory exists
        parent = path.parent
        if not parent.exists():
            print(f"💡 Parent directory doesn't exist: {parent}")
            print(f"   Create it with: mkdir -p {parent}")
        else:
            # Suggest similar files in the directory
            similar_files = list(parent.glob(f"*{path.suffix}")) if path.suffix else list(parent.glob("*"))
            if similar_files:
                print(f"💡 Similar files found in {parent}:")
                for similar_file in similar_files[:3]:  # Show max 3 suggestions
                    print(f"   • {similar_file.name}")
                if len(similar_files) > 3:
                    print(f"   ... and {len(similar_files) - 3} more")
            else:
                print(f"💡 No files found in {parent}")

        return False

    if not path.is_file():
        print(f"❌ Path exists but is not a file: {filepath}")
        return False

    return True


def validate_directory_exists(dirpath: str, create_if_missing: bool = False) -> bool:
    """
    Validate that a directory exists, optionally creating it.

    Args:
        dirpath: Path to the directory
        create_if_missing: Whether to create the directory if it doesn't exist

    Returns:
        True if directory exists/was created, False otherwise
    """
    path = Path(dirpath)

    if not path.exists():
        if create_if_missing:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"📁 Created directory: {dirpath}")
                return True
            except OSError as e:
                print(f"❌ Failed to create directory {dirpath}: {e}")
                return False
        else:
            print(f"❌ Directory not found: {dirpath}")
            print(f"💡 Create it with: mkdir -p {dirpath}")
            return False

    if not path.is_dir():
        print(f"❌ Path exists but is not a directory: {dirpath}")
        return False

    return True


def validate_model_compatibility(embeddings_file: str, base_model: str) -> bool:
    """
    Validate model compatibility and provide helpful warnings.

    Args:
        embeddings_file: Path to embeddings file
        base_model: Base model name

    Returns:
        True if validation passes, False if there are issues
    """
    try:
        from .metadata_utils import get_base_model_from_embeddings, validate_text_embeddings_compatibility

        # Get expected model from embeddings
        expected_model = get_base_model_from_embeddings(embeddings_file)

        if expected_model and expected_model != base_model:
            print(f"⚠️  Base model mismatch detected!")
            print(f"   Embeddings were created with: {expected_model}")
            print(f"   You specified: {base_model}")
            print(f"   💡 Using detected model for compatibility: {expected_model}")
            return False

        # Validate text embeddings compatibility
        if not validate_text_embeddings_compatibility(embeddings_file, base_model):
            print(f"⚠️  Text embeddings compatibility could not be verified")
            print(f"   This may cause issues with model creation")

    except Exception as e:
        logger.debug(f"Model compatibility check failed: {e}")
        print(f"⚠️  Could not verify model compatibility: {e}")

    return True


def display_configuration(config: Dict[str, Any], title: str = "Configuration"):
    """
    Display configuration in a formatted way.

    Args:
        config: Configuration dictionary
        title: Title for the configuration display
    """
    print(f"\n🔧 {title}")
    print("=" * (len(title) + 3))

    for key, value in config.items():
        # Format key nicely
        display_key = key.replace('_', ' ').title()

        # Handle different value types
        if isinstance(value, bool):
            display_value = "✓" if value else "✗"
        elif isinstance(value, list):
            display_value = ", ".join(str(v) for v in value)
        elif value is None:
            display_value = "auto-detect"
        else:
            display_value = str(value)

        print(f"   {display_key}: {display_value}")

    print()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_success_summary(operation: str, results: Dict[str, Any]):
    """Print a standardized success summary."""
    print("\n" + "=" * 50)
    print(f"✅ {operation} Completed Successfully!")
    print("=" * 50)

    # Show key results
    for key, value in results.items():
        if key in ['model_path', 'embeddings_path', 'output_file']:
            print(f"📦 Output: {value}")
        elif key == 'num_embeddings':
            print(f"📊 Embeddings generated: {value:,}")
        elif key == 'elapsed_time':
            print(f"⏱️  Total time: {format_duration(value)}")
        elif key == 'file_size':
            print(f"💾 File size: {format_file_size(value)}")

    print("=" * 50)


def handle_cli_error(error: Exception, operation: str, verbose: bool = False) -> int:
    """
    Handle CLI errors in a standardized way.

    Args:
        error: The exception that occurred
        operation: Description of the operation that failed
        verbose: Whether to show full traceback

    Returns:
        Exit code (1)
    """
    print(f"❌ {operation} failed: {error}")

    if verbose:
        import traceback
        print("\n🔍 Full traceback:")
        print("-" * 40)
        traceback.print_exc()
        print("-" * 40)
    else:
        print("💡 Use --verbose for detailed error information")

    # Provide common troubleshooting tips
    error_str = str(error).lower()

    if "no such file" in error_str or "not found" in error_str:
        print("💡 Troubleshooting tips:")
        print("   • Check that all file paths are correct")
        print("   • Ensure files exist and are readable")
        print("   • Use absolute paths if relative paths fail")

    elif "permission" in error_str:
        print("💡 Troubleshooting tips:")
        print("   • Check file/directory permissions")
        print("   • Ensure you have write access to output directories")

    elif "memory" in error_str or "out of memory" in error_str:
        print("💡 Troubleshooting tips:")
        print("   • Try reducing batch size or model dimensions")
        print("   • Close other memory-intensive applications")
        print("   • Use a machine with more RAM for large ontologies")

    elif "cuda" in error_str or "gpu" in error_str:
        print("💡 Troubleshooting tips:")
        print("   • Check CUDA installation and GPU availability")
        print("   • Try running without GPU acceleration")

    return 1


def with_progress_tracking(operation: str, func: Callable, *args, **kwargs) -> Any:
    """
    Wrapper to add progress tracking to any operation.

    Args:
        operation: Description of the operation
        func: Function to execute
        *args, **kwargs: Arguments for the function

    Returns:
        Function result
    """
    tracker = ProgressTracker(operation)
    tracker.start()

    try:
        result = func(*args, **kwargs)
        tracker.finish(success=True)
        return result
    except Exception as e:
        tracker.finish(success=False, message=str(e))
        raise


def ask_user_confirmation(message: str, default: bool = True) -> bool:
    """
    Ask user for confirmation with a default option.

    Args:
        message: Question to ask
        default: Default answer if user just presses enter

    Returns:
        True if user confirms, False otherwise
    """
    suffix = "[Y/n]" if default else "[y/N]"
    response = input(f"❓ {message} {suffix}: ").lower().strip()

    if not response:
        return default

    return response in ['y', 'yes', 'true', '1']


def print_command_examples(command: str, examples: List[Dict[str, str]]):
    """
    Print command examples in a formatted way.

    Args:
        command: Base command name
        examples: List of dicts with 'description' and 'command' keys
    """
    print(f"\n📝 Examples for 'on2vec {command}':")
    print("-" * 40)

    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['description']}")
        print(f"   on2vec {command} {example['command']}")
        if i < len(examples):
            print()


# Emoji constants for consistent usage
class Emoji:
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    INFO = "💡"
    PROGRESS = "⏳"
    CONFIG = "🔧"
    FILE = "📁"
    OUTPUT = "📦"
    TIME = "⏱️"
    SIZE = "💾"
    STATS = "📊"
    QUESTION = "❓"
    EXAMPLES = "📝"
    SEARCH = "🔍"
    ROCKET = "🚀"
    GEAR = "⚙️"