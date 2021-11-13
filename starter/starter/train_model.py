from pipeline_components import PrepModel, ScoreModel
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creating model assets for the project including model performance metrics"
    )

    parser.add_argument("--input_path_train", type=str, help="Path to the cleaned training dataset")

    parser.add_argument("--input_path_test", type=str, help="Path to the cleaned testing dataset")

    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the directory where the model should be stored",
    )

    parser.add_argument(
        "--metrics_path",
        type=str,
        help="Path to the directory where the metrics should be stored",
    )

    parser.add_argument(
        "--pipe_path",
        type=str,
        help="Path to your fitted data pipeline",
        required=True,
    )

    parser.add_argument(
        "--params_path",
        type=str,
        help="Path to your params.yaml created for DVC",
        default="params.yaml",
        required=False,
    )

    parser.add_argument(
        "--variable",
        type=str,
        help="Name of the variable you want to generate performance slices for",
        required=False,
        default=None,
    )

    args = parser.parse_args()

    PrepModel(
        args.input_path_train, args.output_path, args.params_path, args.pipe_path
    ).model_training()
    
    ScoreModel(
        args.input_path_test,
        args.metrics_path,
        args.params_path,
        args.output_path + '/model.joblib',
        args.variable,
    ).performance_output()