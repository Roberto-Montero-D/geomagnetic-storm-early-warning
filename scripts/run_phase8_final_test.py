"""ONE-TIME Phase 8 protected Final Test runner.

Do not execute this script until its implementation commit is pushed and the
repository worktree is clean.
"""
from __future__ import annotations
import argparse, subprocess
from pathlib import Path

from scripts.run_phase5_screening import build_phase5_screening_inputs
from src.data.kp import build_kp_intervals
from src.data.omni import load_omni
from src.dataset.row_status import build_row_status
from src.final_test.materialization import materialize_phase8_data
from src.final_test.prediction import generate_phase8_predictions
from src.final_test.scoring import score_phase8_final_test, write_phase8_final_artifacts

def _require_clean_main(repo_root: Path):
    branch=subprocess.run(["git","branch","--show-current"],cwd=repo_root,
        check=True,text=True,capture_output=True).stdout.strip()
    status=subprocess.run(["git","status","--porcelain"],cwd=repo_root,
        check=True,text=True,capture_output=True).stdout.strip()
    if branch!="main": raise RuntimeError("Final Test execution requires branch main.")
    if status: raise RuntimeError("Final Test execution requires a clean Git worktree.")

def run_phase8_final_test(fmt_path: Path,lst_path: Path,output_dir: Path,repo_root: Path):
    _require_clean_main(repo_root)
    print("[Phase 8 Final 1/4] Building canonical inputs...",flush=True)
    dataset,splits,_,events=build_phase5_screening_inputs(fmt_path,lst_path)
    status=build_row_status(dataset)
    print("[Phase 8 Final 2/4] Materializing frozen train/test matrices...",flush=True)
    materialized=materialize_phase8_data(dataset,status,splits)
    print("[Phase 8 Final 3/4] Fitting frozen model and generating protected probabilities...",flush=True)
    predictions=generate_phase8_predictions(materialized,progress=True)
    targets=dataset.loc[predictions.table.index,"target"].copy()
    print("[Phase 8 Final 4/4] Applying frozen threshold and scoring ONCE...",flush=True)
    episodes,metrics=score_phase8_final_test(
        predictions.table["probability"],targets,events)
    write_phase8_final_artifacts(output_dir,predictions.table["probability"],
        targets,episodes,metrics)
    return metrics

def parse_args():
    p=argparse.ArgumentParser(description="Execute the ONE-TIME frozen Phase 8 protected Final Test.")
    p.add_argument("--omni-fmt",required=True,type=Path)
    p.add_argument("--omni-lst",required=True,type=Path)
    p.add_argument("--output-dir",type=Path,default=Path("results/phase8/final_test"))
    p.add_argument("--repo-root",type=Path,default=Path("."))
    p.add_argument("--execute-protected-final-test",action="store_true",
        help="Required explicit authorization flag.")
    return p.parse_args()

def main():
    a=parse_args()
    if not a.execute_protected_final_test:
        raise SystemExit("REFUSED: pass --execute-protected-final-test only after the Phase 8.4 implementation commit is pushed.")
    m=run_phase8_final_test(a.omni_fmt,a.omni_lst,a.output_dir,a.repo_root.resolve())
    print("\nPHASE 8 PROTECTED FINAL TEST COMPLETE — DO NOT RETUNE")
    for k,v in m.__dict__.items(): print(f"{k}: {v}")
    print(f"Artifacts: {a.output_dir}")

if __name__=="__main__": main()
