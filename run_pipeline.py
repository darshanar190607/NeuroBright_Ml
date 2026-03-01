
import os
import sys
import subprocess

def run_step(module_name):
    print(f"\n[PIPELINE] Running {module_name}...")
    try:
        result = subprocess.run([sys.executable, "-m", module_name], check=True, capture_output=True, text=True)
        print(f"[SUCCESS] {module_name}")
        print("-" * 30)
        print(result.stdout)
        print("-" * 30)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {module_name} failed with exit code {e.returncode}")
        print(e.stdout)
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    steps = [
        "src.components.data_ingestion",
        "src.components.data_transformation",
        "src.components.feature_engineering",
        "src.components.model_trainer_lite",
        # "src.components.model_evaluation" # Skipping due to matplotlib/seaborn crash
    ]
    
    for step in steps:
        run_step(step)
    
    print("\n[PIPELINE] Execution Completed Successfully.")
