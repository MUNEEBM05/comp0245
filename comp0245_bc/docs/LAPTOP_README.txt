SUPER TEST - LAPTOP HALF (David, 1 Nov)
=======================================
Runs 2 of the 5 systems: (S1) expert arm + A* planner, (S5) camera version.
The server runs the other 3. About 2-3 hours on a Ryzen 7.

1. Unzip this folder anywhere, e.g. C:\comp0245_super
2. Plug the laptop in. Settings > System > Power: set "sleep" to NEVER while it runs.
3. Link the robot models (PowerShell, inside the unzipped folder, once):
     New-Item -ItemType Junction -Path .\models -Target "C:\Users\YOU\...\RoboEnv\models"
   (the same models folder your other COMP0245 code uses)
4. Open the RoboEnv pixi shell, go to the unzipped folder, and run:
     cd C:\Users\YOU\...\RoboEnv
     pixi shell
     cd C:\comp0245_super
     python run_super_laptop.py
5. It prints one line per test. If it stops, just run the same command again: it skips finished parts.
6. When it says "== done", zip the results_super folder and send it to Claude.

Quick check first (2 minutes): python run_super_laptop.py --workers 7
If the first line after "1 fresh ..." shows an error instead of "success ...%", send a screenshot.
