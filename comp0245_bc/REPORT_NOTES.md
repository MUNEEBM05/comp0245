# Report notes – decisions and findings to write up

Short notes for the report draft (due Wed 4 Nov). One section per decision: what we found, what we did, the sentence to use.

## 1. Input ablation: we kept all 71 inputs

- **What we tested:** removed one input group at a time, retrained, and drove 50 benchmark scenes (held 5 s within 1 cm).

  | Removed | Held 5 s |
  |---|---|
  | Nothing (model R) | 92% |
  | IK averages | 92% |
  | Error averages | 90% |
  | History | 60% |
  | IK direction now | 58% |
  | Goal (parking spot) | 46% |

- **Decision:** keep all 71 inputs in model R. We did not retrain a "lean R" without the averages.
- **Why:** removing them gains little (71 → 65 inputs; R is already well under 1 ms per decision). It would also mean redoing the 3 seeds, the fresh test and Stage B (about 4–5 h) just before the deadline, for no change in success.
- **Report sentence:** "An ablation showed the goal, the IK direction and the history inputs carry the performance, while the running averages are redundant (92% and 90% without them); we kept them, since removing them would not change success and would require re-validating the final model."
- **Possible extra if time allows:** train a lean R overnight and report it as a side result.

## 2. Unfinished experiments in the code

- **Experiments:** DAgger (`dagger.py`, `markov_expert.py`), GRU (`train_gru.py`), TCN/attention (`train_conv.py`), noise training (`augment.py`).
- **Decision:** keep them, but move them into an `experiments/` folder so the main folder holds only the core pipeline (Abdullah's clean-up). Their results stay in the report as "tried, did not beat R".

## 4. Objects and obstacles (Valerio's suggestion, added 30 Oct)

- **Tests:** all on the same 50 tier-1 scenes, expert vs R (+20 ms filter), held 5 s within 1 cm (`run_objects.sh`, `--set objects=K`).
  1. An object at the target (cube on a stand).
  2. Clutter not in the way (5 ready-made PyBullet objects).
  3. One box in the path.
  4. (planned) A wall with a 1.0 m door, so the target is hidden behind it.
- **Expected story:** objects and clutter do not matter, because the model only gets the target point. Both the expert and R crash into the box and the wall: a copy cannot know more than its teacher (the expert has no obstacle sensing). This is the limitation of behaviour cloning.
- **Going beyond the expert (planned):** keep R and add a classical navigation layer on top.
  - LiDAR: a ring of PyBullet rays from the base.
  - VFH: steer through the open gap nearest the goal direction; gets through narrow doors without the potential-field repulsion problem.
  - Bug 2: if no gap points toward the goal, follow the obstacle edge until back on the start-goal line closer to the goal; guarantees a way round a hidden goal.
  - R takes over again to park and reach. The arm is never touched.
- **Tests for it:** R+nav on tests 3 and 4, plus R+nav on normal scenes, to show the layer does not hurt the normal case.
- **Scene details (fair for every policy):** in tests 3 and 4 the robot starts 2.5 m from its parking spot (same direction and heading as the original scene), so the obstacle is always really in the way. Before this, some tier-1 starts were only 0.1–0.5 m from the spot. Those scenes get a 60 s limit instead of 40 s, because a detour takes longer. The door is 1.4 m wide (robot 0.62 m wide), 1.0 m to one side of the straight path, in a 6 m wall.
- **What we learned building it (good report material):** the base has almost no sideways grip (wheel lateral friction 0.01 in the template). If it turns while moving, it keeps gliding sideways like on ice. This is also why the expert sometimes orbits the parking spot. So the layer:
  - stops, turns on the spot, then drives dead straight;
  - turns only when nothing is within arm reach (0.9 m) in the arc the arm sweeps through, otherwise backs up a little first, because the folded arm sticks out 0.75 m in front;
  - holds the arm folded in its start pose during the detour (PD + gravity, like the template), because the learned arm sees states it never trained on;
  - drives to the parking spot, faces the target, then hands everything back to R for the final parking and the reach.
- **Safety corridor:** a heading is free if no LiDAR point lies within 0.65 m of the straight line ahead (1.5 m look-ahead). This is the exact check for a round robot, so a door wider than 1.3 m shows as open. The Bug 2 side is the one with the open direction (LiDAR sees nothing within 3.5 m, e.g. the doorway) closest to the goal.
- **Honest note:** the navigation layer is hand-written, not learned. The proper long-term fix is a planning expert with LiDAR, cloned again (future work). SLAM is not needed: the simulator gives the robot's position.
- **Report sentence:** "Behaviour cloning inherits its teacher's blindness: both crash into unseen obstacles. Adding a LiDAR with VFH steering and a Bug 2 fallback lets the learned controller go beyond its teacher, while the learned skill still does the parking and reaching."

## 3. Bayesian optimisation (Lab 3, skopt)

- **Setup:** 12 trials on 20 separate tuning scenes, scored by held 5 s clean + with sensor noise.
- **Result:** the best was a 3×192 network, batch 2048, filter 29 ms (95% / 95%), against R's settings (85% / 80%).
- **Caveat:** one scene = 5%. Near-identical settings scored 85% / 85%, so the gain is within noise. This matches Valerio's warning that Bayesian optimisation on an MLP is tricky: the score is noisy because of random training starts, each trial is expensive, and only a handful of settings can be tuned, never the weights.
- **Report sentence:** "Bayesian optimisation suggested a smaller network and a longer position filter, but the gain was within the noise of 20 tuning scenes, so we kept model R and adopted the longer (20 ms) filter."
