import os
import subprocess
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_owl_files(directory, output_image_dir, output_parquet_dir):
    # Ensure output directories exist
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
        logger.info(f"Created output directory for images: {output_image_dir}")
    
    if not os.path.exists(output_parquet_dir):
        os.makedirs(output_parquet_dir)
        logger.info(f"Created output directory for Parquet files: {output_parquet_dir}")

    # Loop through files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".owl"):
            owl_file = os.path.join(directory, filename)
            output_image = os.path.join(output_image_dir, f"{os.path.splitext(filename)[0]}.png")
            output_parquet = os.path.join(output_parquet_dir, f"{os.path.splitext(filename)[0]}.parquet")
            logger.info(f"Processing OWL file: {owl_file}")

            try:
                # Run force_layout.py on each OWL file, including Parquet output
                subprocess.run(
                    ['python3', 'force_layout.py', owl_file, '--output_image', output_image, '--output_parquet', output_parquet],
                    check=True
                )
                logger.info(f"Successfully processed {owl_file} and saved image to {output_image} and Parquet to {output_parquet}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error processing {owl_file}: {e}")
        else:
            logger.info(f"Skipping non-OWL file: {filename}")

def main():
    # Argument parser for input directory and output directories
    parser = argparse.ArgumentParser(description="Process a directory of OWL files and generate force-directed layouts and Parquet coordinates")
    parser.add_argument('directory', help="Path to the directory containing OWL files")
    parser.add_argument('output_image_dir', help="Path to the output directory to save images")
    parser.add_argument('output_parquet_dir', help="Path to the output directory to save Parquet files")
    args = parser.parse_args()

    # Process all OWL files in the directory
    process_owl_files(args.directory, args.output_image_dir, args.output_parquet_dir)

if __name__ == "__main__":
    main()
