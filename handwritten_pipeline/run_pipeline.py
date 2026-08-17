#!/usr/bin/env python
import os
import argparse
import sys
import logging

from src.batch_runner import BatchRunner
from src.evaluation import run_evaluation_metrics

def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Handwritten Answer Sheet Evaluation Pipeline"
    )
    parser.add_argument(
        "--config",
        default=os.path.join("configs", "pipeline_config.yaml"),
        help="Path to the YAML configuration file (default: configs/pipeline_config.yaml)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at: {args.config}")
        sys.exit(1)

    print("\n-------------------------------------------------------------")
    print("  Smart Answer Evaluation - Handwritten Extension Pipeline  ")
    print("-------------------------------------------------------------\n")

    try:
        # Initialize and run batch runner
        runner = BatchRunner(config_path=args.config)
        results = runner.run()

        # Run quantitative evaluations if enabled
        if runner.grading_settings.get("run_evaluation", True) and results:
            print("\n-------------------------------------------------------------")
            print("  Running Quantitative Evaluation & Generating Reports...  ")
            print("-------------------------------------------------------------\n")
            
            output_dir = runner.paths.get("output_dir", "output_handwritten")
            gt_csv = runner.grading_settings.get("ground_truth_csv")
            
            run_evaluation_metrics(
                results_csv_path=runner.checkpoint_path,
                output_dir=output_dir,
                ground_truth_csv_path=gt_csv
            )
            print(f"\nAll evaluations complete. Outputs saved in directory: {output_dir}")
        else:
            print("\nBatch runner complete. Evaluations skipped.")

    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline failed: {e}")
        logging.getLogger("HandwrittenPipeline").critical("Pipeline crashed", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
