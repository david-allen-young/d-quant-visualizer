import argparse
import subprocess
import os
import json

def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run full expression analysis pipeline")
    parser.add_argument("--midi_path", required=True, help="Path to training MIDI file")
    parser.add_argument("--category", required=True, help="Category name (e.g., crescendo, diminuendo, stable)")
    parser.add_argument("--count", type=int, default=100, help="How many morphs to generate")
    args = parser.parse_args()

    # Derived folder structure
    base = "assets"
    env_dir = os.path.join(base, "envelopes_csv", args.category)
    morph_dir = os.path.join(base, "morph_csv", args.category)
    morph_gen_dir = os.path.join(morph_dir, "generated")

    os.makedirs(env_dir, exist_ok=True)
    os.makedirs(morph_dir, exist_ok=True)
    os.makedirs(morph_gen_dir, exist_ok=True)

    # Step 1: Create config for analyze_dynamics and run it
    pipeline_cfg = {
        "dynamizer_midi_path": args.midi_path,
        "envelope_csv_dir": env_dir
    }
    with open("pipeline_config.json", "w") as f:
        json.dump(pipeline_cfg, f, indent=2)

    run_cmd(["analyze_dynamics.exe"])

    # Step 2: Analyze envelopes (generate mean/std)
    run_cmd([
        "python", "generate_envelope.py", "analyze",
        "--csv_dir", env_dir,
        "--output_dir", morph_dir
    ])

    # Step 3: Generate morph2 variations
    run_cmd([
        "python", "generate_envelope.py", "generate",
        "--method", "morph2",
        "--mean_path", os.path.join(morph_dir, "mean_envelope.npy"),
        "--std_path", os.path.join(morph_dir, "std_envelope.npy"),
        "--input_csv_dir", env_dir,
        "--count", str(args.count),
        "--save_dir", morph_gen_dir
    ])

    # Step 4: Visualize final output
    run_cmd([
        "python", "visualize_variation.py",
        "--input_dir", morph_gen_dir
    ])

if __name__ == "__main__":
    main()
