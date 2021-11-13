import argparse
from pipeline_components import PrepData

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Creating data assets for the project")

    parser.add_argument("--input_path", type=str, help="Path to the raw dataset")

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the directory where the assets should be stored",
    )

    parser.add_argument(
        "--params_path",
        type=str,
        help="Path to your params.yaml created for DVC",
        default="params.yaml",
        required=False,
    )

    args = parser.parse_args()

    PrepData(args.input_path, args.output_path, args.params_path).preprocess()
