"""Shared paths for every code folder. Each code folder has an identical copy of this file.

ROOT is the project folder (comp0245_bc), where trained/, data/, benchmark/ and results/ live.
Importing this file also puts every code folder on the import path, so modules in different
folders still import each other by plain name (from rollout import make_policy), as before.
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_FOLDERS = ("core", "navigation", "training", "noise_and_delay", "evaluation", "analysis")
for _d in CODE_FOLDERS:
    _p = os.path.join(ROOT, _d)
    if _p not in sys.path:
        sys.path.append(_p)

# Result folders that hold other test sets under the same file names as the 50-scene benchmark
# (10-scene first tests, Muneeb's K diagnosis, BO tuning scenes, rejected navigation versions,
# the 120 s door check, fresh scenes, super test). find_result only looks inside them when asked directly.
SEPARATE = ("first_10_scenes", "K_diagnosis", "bayes_opt_trials", "rejected_navigation", "door_120s", "3_final_tests", "abdullah_model_C", "abdullah_bases")


def find_result(folder, name):
    """Path of a results file: folder/name if it is there, else the match in folder's subfolders.

    Results are sorted by project section and system (results/<section>/.../<system>/), so a file is
    found by its name alone. Subfolders listed in SEPARATE are skipped, so find_result(results/, name)
    returns the 50-scene benchmark result.
    """
    path = os.path.join(folder, name)
    if os.path.exists(path):
        return path
    for hit in sorted(glob.glob(os.path.join(folder, "**", name), recursive=True)):
        if not set(os.path.relpath(hit, folder).split(os.sep)) & set(SEPARATE):
            return hit
    return path
