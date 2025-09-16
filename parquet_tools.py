#!/usr/bin/env python3
"""
CLI tool for working with on2vec embedding Parquet files.
Provides utilities for inspecting metadata, converting formats, and vector operations.
"""

import argparse
import logging
import sys
from on2vec.io import (
    inspect_parquet_metadata,
    convert_parquet_to_csv,
    load_embeddings_as_dataframe,
    add_embedding_vectors,
    subtract_embedding_vectors,
    get_embedding_vector
)
from on2vec.visualization import (
    plot_pca_2d,
    plot_tsne_2d,
    plot_umap_2d,
    plot_embedding_distribution,
    create_visualization_comparison
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def cmd_inspect(args):
    """Inspect metadata of a Parquet file."""
    try:
        metadata = inspect_parquet_metadata(args.parquet_file)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_convert(args):
    """Convert Parquet file to CSV."""
    try:
        output_file = convert_parquet_to_csv(args.parquet_file, args.output)
        print(f"✅ Converted to CSV: {output_file}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_list(args):
    """List all node IDs in a Parquet file."""
    try:
        df = load_embeddings_as_dataframe(args.parquet_file)
        node_ids = df['node_id'].to_list()

        print(f"\n📋 Node IDs in {args.parquet_file}")
        print("=" * 60)

        if args.limit:
            if len(node_ids) > args.limit:
                print(f"Showing first {args.limit} of {len(node_ids)} total node IDs:")
                node_ids = node_ids[:args.limit]

        for i, node_id in enumerate(node_ids, 1):
            if args.numbered:
                print(f"{i:4d}. {node_id}")
            else:
                print(node_id)

        if args.limit and len(df) > args.limit:
            print(f"\n... and {len(df) - args.limit} more")

        print(f"\nTotal: {len(df)} embeddings")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_get(args):
    """Get a specific embedding vector."""
    try:
        vector = get_embedding_vector(args.parquet_file, args.node_id)

        print(f"\n🔍 Embedding for: {args.node_id}")
        print("=" * 60)
        print(f"Dimensions: {len(vector)}")

        if args.format == 'array':
            print(f"Vector: {vector}")
        elif args.format == 'list':
            print("Vector:")
            for i, val in enumerate(vector):
                print(f"  dim_{i}: {val}")
        elif args.format == 'json':
            import json
            result = {
                'node_id': args.node_id,
                'dimensions': len(vector),
                'vector': vector.tolist()
            }
            print(json.dumps(result, indent=2))

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_add(args):
    """Add two embedding vectors."""
    try:
        # Handle case where both vectors are from the same file
        file2 = args.parquet_file2 if args.parquet_file2 else args.parquet_file1

        result = add_embedding_vectors(args.parquet_file1, args.node_id1, file2, args.node_id2)

        print(f"\n➕ Vector Addition Result")
        print("=" * 60)
        print(f"{args.node_id1} + {args.node_id2}")
        print(f"Result dimensions: {len(result)}")

        if args.format == 'array':
            print(f"Result: {result}")
        elif args.format == 'list':
            print("Result:")
            for i, val in enumerate(result):
                print(f"  dim_{i}: {val}")
        elif args.format == 'json':
            import json
            result_data = {
                'operation': 'addition',
                'operand1': args.node_id1,
                'operand2': args.node_id2,
                'dimensions': len(result),
                'result': result.tolist()
            }
            print(json.dumps(result_data, indent=2))

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_subtract(args):
    """Subtract two embedding vectors."""
    try:
        # Handle case where both vectors are from the same file
        file2 = args.parquet_file2 if args.parquet_file2 else args.parquet_file1

        result = subtract_embedding_vectors(args.parquet_file1, args.node_id1, file2, args.node_id2)

        print(f"\n➖ Vector Subtraction Result")
        print("=" * 60)
        print(f"{args.node_id1} - {args.node_id2}")
        print(f"Result dimensions: {len(result)}")

        if args.format == 'array':
            print(f"Result: {result}")
        elif args.format == 'list':
            print("Result:")
            for i, val in enumerate(result):
                print(f"  dim_{i}: {val}")
        elif args.format == 'json':
            import json
            result_data = {
                'operation': 'subtraction',
                'operand1': args.node_id1,
                'operand2': args.node_id2,
                'dimensions': len(result),
                'result': result.tolist()
            }
            print(json.dumps(result_data, indent=2))

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_plot_pca(args):
    """Create a 2D PCA plot of embeddings."""
    try:
        import matplotlib.pyplot as plt

        fig, ax, pca_embeddings = plot_pca_2d(
            parquet_file=args.parquet_file,
            output_file=args.output,
            figsize=(args.width, args.height),
            alpha=args.alpha,
            s=args.point_size,
            random_state=args.seed
        )

        if not args.output:
            print(f"📊 PCA plot created with {len(pca_embeddings)} points")
            if args.show:
                plt.show()
        else:
            print(f"📊 PCA plot saved to {args.output}")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_plot_tsne(args):
    """Create a 2D t-SNE plot of embeddings."""
    try:
        import matplotlib.pyplot as plt

        fig, ax, tsne_embeddings = plot_tsne_2d(
            parquet_file=args.parquet_file,
            output_file=args.output,
            figsize=(args.width, args.height),
            alpha=args.alpha,
            s=args.point_size,
            perplexity=args.perplexity,
            random_state=args.seed,
            max_iter=args.max_iter
        )

        if not args.output:
            print(f"📊 t-SNE plot created with {len(tsne_embeddings)} points")
            if args.show:
                plt.show()
        else:
            print(f"📊 t-SNE plot saved to {args.output}")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_plot_umap(args):
    """Create a 2D UMAP plot of embeddings."""
    try:
        import matplotlib.pyplot as plt

        fig, ax, umap_embeddings = plot_umap_2d(
            parquet_file=args.parquet_file,
            output_file=args.output,
            figsize=(args.width, args.height),
            alpha=args.alpha,
            s=args.point_size,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.seed
        )

        if not args.output:
            print(f"📊 UMAP plot created with {len(umap_embeddings)} points")
            if args.show:
                plt.show()
        else:
            print(f"📊 UMAP plot saved to {args.output}")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_plot_distribution(args):
    """Create distribution plots for embedding dimensions."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plot_embedding_distribution(
            parquet_file=args.parquet_file,
            output_file=args.output,
            figsize=(args.width, args.height)
        )

        if not args.output:
            print("📊 Distribution plot created")
            if args.show:
                plt.show()
        else:
            print(f"📊 Distribution plot saved to {args.output}")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_plot_compare(args):
    """Create comparison plot showing PCA, t-SNE, and UMAP side by side."""
    try:
        import matplotlib.pyplot as plt

        fig, axes, results = create_visualization_comparison(
            parquet_file=args.parquet_file,
            output_file=args.output,
            figsize=(args.width, args.height),
            random_state=args.seed
        )

        if not args.output:
            print("📊 Comparison plot created")
            if args.show:
                plt.show()
        else:
            print(f"📊 Comparison plot saved to {args.output}")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool for working with on2vec embedding Parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect metadata
  python parquet_tools.py inspect embeddings.parquet

  # Convert to CSV
  python parquet_tools.py convert embeddings.parquet --output embeddings.csv

  # List all node IDs
  python parquet_tools.py list embeddings.parquet

  # Get specific embedding
  python parquet_tools.py get embeddings.parquet "http://example.org/Class1"

  # Add two vectors from same file
  python parquet_tools.py add embeddings.parquet "http://example.org/Class1" "http://example.org/Class2"

  # Subtract vectors from different files
  python parquet_tools.py subtract file1.parquet "Class1" file2.parquet "Class2"
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    subparsers.required = True

    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Display metadata from Parquet file')
    inspect_parser.add_argument('parquet_file', help='Path to Parquet file')
    inspect_parser.set_defaults(func=cmd_inspect)

    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert Parquet to CSV')
    convert_parser.add_argument('parquet_file', help='Path to input Parquet file')
    convert_parser.add_argument('--output', '-o', help='Output CSV file path')
    convert_parser.set_defaults(func=cmd_convert)

    # List command
    list_parser = subparsers.add_parser('list', help='List node IDs in Parquet file')
    list_parser.add_argument('parquet_file', help='Path to Parquet file')
    list_parser.add_argument('--limit', '-n', type=int, help='Limit number of IDs shown')
    list_parser.add_argument('--numbered', action='store_true', help='Show line numbers')
    list_parser.set_defaults(func=cmd_list)

    # Get command
    get_parser = subparsers.add_parser('get', help='Get specific embedding vector')
    get_parser.add_argument('parquet_file', help='Path to Parquet file')
    get_parser.add_argument('node_id', help='Node ID to retrieve')
    get_parser.add_argument('--format', choices=['array', 'list', 'json'], default='array',
                           help='Output format')
    get_parser.set_defaults(func=cmd_get)

    # Add command
    add_parser = subparsers.add_parser('add', help='Add two embedding vectors')
    add_parser.add_argument('parquet_file1', help='Path to first Parquet file')
    add_parser.add_argument('node_id1', help='Node ID in first file')
    add_parser.add_argument('node_id2', help='Node ID in second file (or same file)')
    add_parser.add_argument('parquet_file2', nargs='?', help='Path to second Parquet file (optional)')
    add_parser.add_argument('--format', choices=['array', 'list', 'json'], default='array',
                           help='Output format')
    add_parser.set_defaults(func=cmd_add)

    # Subtract command
    subtract_parser = subparsers.add_parser('subtract', help='Subtract two embedding vectors')
    subtract_parser.add_argument('parquet_file1', help='Path to first Parquet file (minuend)')
    subtract_parser.add_argument('node_id1', help='Node ID in first file')
    subtract_parser.add_argument('node_id2', help='Node ID in second file (or same file)')
    subtract_parser.add_argument('parquet_file2', nargs='?', help='Path to second Parquet file (optional)')
    subtract_parser.add_argument('--format', choices=['array', 'list', 'json'], default='array',
                                help='Output format')
    subtract_parser.set_defaults(func=cmd_subtract)

    # Plot PCA command
    pca_parser = subparsers.add_parser('plot-pca', help='Create 2D PCA plot of embeddings')
    pca_parser.add_argument('parquet_file', help='Path to Parquet file')
    pca_parser.add_argument('--output', '-o', help='Output image file path (PNG, PDF, etc.)')
    pca_parser.add_argument('--width', type=float, default=12, help='Plot width in inches')
    pca_parser.add_argument('--height', type=float, default=8, help='Plot height in inches')
    pca_parser.add_argument('--alpha', type=float, default=0.6, help='Point transparency (0-1)')
    pca_parser.add_argument('--point-size', type=float, default=20, help='Point size')
    pca_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    pca_parser.add_argument('--show', action='store_true', help='Display plot interactively')
    pca_parser.set_defaults(func=cmd_plot_pca)

    # Plot t-SNE command
    tsne_parser = subparsers.add_parser('plot-tsne', help='Create 2D t-SNE plot of embeddings')
    tsne_parser.add_argument('parquet_file', help='Path to Parquet file')
    tsne_parser.add_argument('--output', '-o', help='Output image file path (PNG, PDF, etc.)')
    tsne_parser.add_argument('--width', type=float, default=12, help='Plot width in inches')
    tsne_parser.add_argument('--height', type=float, default=8, help='Plot height in inches')
    tsne_parser.add_argument('--alpha', type=float, default=0.6, help='Point transparency (0-1)')
    tsne_parser.add_argument('--point-size', type=float, default=20, help='Point size')
    tsne_parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity parameter')
    tsne_parser.add_argument('--max-iter', type=int, default=1000, help='Maximum iterations')
    tsne_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    tsne_parser.add_argument('--show', action='store_true', help='Display plot interactively')
    tsne_parser.set_defaults(func=cmd_plot_tsne)

    # Plot UMAP command
    umap_parser = subparsers.add_parser('plot-umap', help='Create 2D UMAP plot of embeddings')
    umap_parser.add_argument('parquet_file', help='Path to Parquet file')
    umap_parser.add_argument('--output', '-o', help='Output image file path (PNG, PDF, etc.)')
    umap_parser.add_argument('--width', type=float, default=12, help='Plot width in inches')
    umap_parser.add_argument('--height', type=float, default=8, help='Plot height in inches')
    umap_parser.add_argument('--alpha', type=float, default=0.6, help='Point transparency (0-1)')
    umap_parser.add_argument('--point-size', type=float, default=20, help='Point size')
    umap_parser.add_argument('--n-neighbors', type=int, default=15, help='UMAP n_neighbors parameter')
    umap_parser.add_argument('--min-dist', type=float, default=0.1, help='UMAP min_dist parameter')
    umap_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    umap_parser.add_argument('--show', action='store_true', help='Display plot interactively')
    umap_parser.set_defaults(func=cmd_plot_umap)

    # Plot distribution command
    dist_parser = subparsers.add_parser('plot-dist', help='Create distribution plots for embedding dimensions')
    dist_parser.add_argument('parquet_file', help='Path to Parquet file')
    dist_parser.add_argument('--output', '-o', help='Output image file path (PNG, PDF, etc.)')
    dist_parser.add_argument('--width', type=float, default=15, help='Plot width in inches')
    dist_parser.add_argument('--height', type=float, default=10, help='Plot height in inches')
    dist_parser.add_argument('--show', action='store_true', help='Display plot interactively')
    dist_parser.set_defaults(func=cmd_plot_distribution)

    # Plot comparison command
    compare_parser = subparsers.add_parser('plot-compare', help='Create comparison plot (PCA, t-SNE, UMAP)')
    compare_parser.add_argument('parquet_file', help='Path to Parquet file')
    compare_parser.add_argument('--output', '-o', help='Output image file path (PNG, PDF, etc.)')
    compare_parser.add_argument('--width', type=float, default=18, help='Plot width in inches')
    compare_parser.add_argument('--height', type=float, default=6, help='Plot height in inches')
    compare_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    compare_parser.add_argument('--show', action='store_true', help='Display plot interactively')
    compare_parser.set_defaults(func=cmd_plot_compare)

    args = parser.parse_args()

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())