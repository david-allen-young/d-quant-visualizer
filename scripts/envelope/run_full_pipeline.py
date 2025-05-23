import argparse
import subprocess
import os
import json

import sys
python_bin = sys.executable

def run_cmd(cmd, cwd=None):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)

def load_config_if_no_args(script_dir):
    config_file = os.path.join(script_dir, "pipeline_wrapper_config.json")
    print(f"[INFO] Checking for config: {config_file}")
    if os.path.exists(config_file):
        print(f"[INFO] Found config file: {config_file}")
        with open(config_file, "r") as f:
            data = json.load(f)
            print(f"[INFO] Loaded config: {json.dumps(data, indent=2)}")
            return data
    print("[INFO] No config file found.")
    return None

def find_repo_root(script_dir, target_folder_name="d-quant"):
    current = script_dir
    while True:
        candidate = os.path.join(current, target_folder_name)
        if os.path.isdir(candidate):
            return os.path.normpath(candidate)
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError(f"Could not locate {target_folder_name} folder from {script_dir}")
        current = parent

def find_sibling_repo(script_dir, sibling_name="d-quant"):
    current = os.path.abspath(script_dir)
    while True:
        parent = os.path.dirname(current)
        candidate = os.path.join(parent, sibling_name)
        print(f"[DEBUG] Trying: {candidate}")
        if os.path.isdir(candidate):
            return os.path.normpath(candidate)
        if parent == current:
            break  # root reached
        current = parent
    raise RuntimeError(f"Could not find sibling folder '{sibling_name}' relative to {script_dir}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[INFO] Current working directory: {os.getcwd()}")
    print(f"[INFO] Script location: {script_dir}")

    parser = argparse.ArgumentParser(description="Run full expression analysis pipeline")
    parser.add_argument("--midi_path", help="Path to training MIDI file")
    parser.add_argument("--category", help="Category name (e.g., crescendo, diminuendo, stable)")
    parser.add_argument("--count", type=int, default=100, help="How many morphs to generate")
    parser.add_argument("--analyze_exe", help="Path to analyze_dynamics.exe")
    args = parser.parse_args()

    print("[INFO] Parsed command-line args:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    if not args.midi_path and not args.category and not args.analyze_exe:
        print("[INFO] No required CLI args provided. Attempting to load config file.")
        config = load_config_if_no_args(script_dir)
        if not config:
            raise RuntimeError("No arguments provided and no pipeline_wrapper_config.json found.")
        args.midi_path = config["midi_path"]
        args.category = config["category"]
        args.count = config.get("count", 100)
        args.analyze_exe = config.get("analyze_exe", "analyze_dynamics.exe")

    if not all([args.midi_path, args.category, args.analyze_exe]):
        raise ValueError("--midi_path, --category, and --analyze_exe are required.")

    # Derived folder structure relative to script directory
    # base = os.path.join(script_dir, "assets")
    # env_dir = os.path.join(base, "envelopes_csv", args.category)
    # morph_dir = os.path.join(base, "morph_csv", args.category)
    # morph_gen_dir = os.path.join(morph_dir, "generated")
    
    # repo_root = find_repo_root(script_dir)
    # dquant_root = os.path.join(repo_root, "d-quant")
    dquant_root = find_sibling_repo(script_dir, "d-quant")

    dynamizer_training = os.path.normpath(os.path.join(dquant_root, "assets", "training", "dynamizer", args.category))
    dynamizer_analysis = os.path.normpath(os.path.join(dquant_root, "assets", "analysis", "dynamizer", args.category))
    dynamizer_generation = os.path.normpath(os.path.join(dquant_root, "assets", "generation", "dynamizer", args.category))

    print("[DEBUG] dynamizer_training == ", dynamizer_training)

    # os.makedirs(env_dir, exist_ok=True)
    # os.makedirs(morph_dir, exist_ok=True)
    # os.makedirs(morph_gen_dir, exist_ok=True)

    os.makedirs(dynamizer_training, exist_ok=True)
    os.makedirs(dynamizer_analysis, exist_ok=True)
    os.makedirs(dynamizer_generation, exist_ok=True)


    # Step 1: Create config for analyze_dynamics and run it
    pipeline_cfg_path = os.path.normpath(os.path.join(script_dir, "pipeline_config.json"))
    pipeline_cfg = {
        # "dynamizer_midi_path": args.midi_path,
        # "envelope_csv_dir": env_dir
        "dynamizer_midi_path": os.path.normpath(os.path.join(dynamizer_training, args.midi_path)),
        "envelope_csv_dir": dynamizer_analysis
    }
    print(f"[INFO] Writing analyze_dynamics config to {pipeline_cfg_path}")
    with open(pipeline_cfg_path, "w") as f:
        json.dump(pipeline_cfg, f, indent=2)

    analyze_exe_dir = os.path.normpath(os.path.dirname(args.analyze_exe))
    # Old (mixed slashes can cause issues)
    # target_config_path = os.path.join(analyze_exe_dir, "pipeline_config.json")
    # New (safe on Windows and cross-platform)
    target_config_path = os.path.normpath(os.path.join(analyze_exe_dir, "pipeline_config.json"))
    print(f"[INFO] Copying config to: {target_config_path}")
    with open(target_config_path, "w") as f:
        json.dump(pipeline_cfg, f, indent=2)

    run_cmd([args.analyze_exe], cwd=os.path.dirname(args.analyze_exe))



    # Step 2: Analyze envelopes (generate mean/std)
    run_cmd([
        python_bin, os.path.normpath(os.path.join(script_dir, "generate_envelope.py")), "analyze",
        "--csv_dir", dynamizer_analysis,
        "--output_dir", dynamizer_analysis
    ])

    # Step 3: Generate morph2 variations
    run_cmd([
        python_bin, os.path.normpath(os.path.join(script_dir, "generate_envelope.py")), "generate",
        "--method", "morph2",
        "--mean_path", os.path.normpath(os.path.join(dynamizer_analysis, "mean_envelope.npy")),
        "--std_path", os.path.normpath(os.path.join(dynamizer_analysis, "std_envelope.npy")),
        "--input_csv_dir", dynamizer_analysis,
        "--count", str(args.count),
        "--save_dir", dynamizer_generation,
        "--category", args.category,
    ])

    # Step 4: Visualize final output
    run_cmd([
        python_bin, os.path.normpath(os.path.join(script_dir, "visualize_variation.py")),
        "--input_dir", dynamizer_generation
    ])

if __name__ == "__main__":
    main()
