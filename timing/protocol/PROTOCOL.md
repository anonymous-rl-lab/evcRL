# Signal-timing information in fixed-actor smooth execution

Date: 2026-09-12. This protocol is fixed before new closed-loop outcomes.

## Question and intervention

Does enabling a signal countdown in the executor improve signal-window longitudinal smoothness for the existing unregularized actor A, without material time, energy, or safety costs?

Both arms use exactly the same frozen A actor (development seed 7), its unchanged thirteen-channel observation, vehicle model, route, 0.5 s integrator, two-second command update, actual jerk target 2 m/s^3, acceleration bounds [-3.5,2.6] m/s^2, speed limits, brake-release backups, and emergency branch. No training, optimizer steps, new reward, or parameter search occurs.

- `color`: the executor receives current signal color within 1000 m. Red/nonpassable color invokes the common stopping backup; green permits the common actor-command projection without a countdown-based stop-or-clear test.
- `timing`: the executor additionally receives the true time remaining to the phase change within the same 1000 m range. On green it applies the existing stop-or-clear test. On red and outside range its execution is exactly the color arm.

The color arm is a reactive controller, not an all-future-phase safety guarantee. Unknown countdown is not replaced with a fabricated long green, and does not force permanent stopping on green. Always preserving a stop before the line with an arbitrarily imminent phase change would prevent passage; that would be an unsuitable artificially immobilized baseline. Both arms retain current-red stopping, physical limits and the same emergency handling; all failures remain reported.

Signal positions and curve geometry are common static map information. Current signal color and executor countdown both have a 1000 m range. Actor input is unchanged in both arms and already contains countdown within its range: the estimand is the incremental use of timing at the executor interface, not removing timing from the complete controller. The environment and metric logger may read ground truth for physical evaluation, but the pure execution function receives no clock, offset, or hidden phase. Neither arm fits a new policy or uses future realized vehicle states.

## Conditions and completion

Retain the existing 4 km route, 35 km/h curve at 1450–1550 m, signal at 3000 m, cycle 90 s/green 30 s, road cap 80 km/h, and genuine deadline 600 s. Settled completion requires x>=3999 m, v<=1e-6 m/s, |a|<=1e-6 m/s^2. Include the full settling tail, emergency actions, time and energy. Stop on a red crossing or deadline; never interpret an incomplete trip's small jerk or energy as a gain.

Development: existing three packs crossed with offsets 0,30,60 s at initial speed 16 m/s, nine conditions. Smoke uses condition 0 for both arms, wall-clock cap ten minutes. Pilot completes the other eight pairs, cap one hour; completed smoke conditions are reused.

Conditional reporting: the previously unused three packs x initial speeds 14,18 m/s x offsets 15,45,75 s, eighteen conditions, two arms = 36 trips. One actor from one seed; these are condition-level paired responses, not 36 seeds or an external-distribution test. Reporting is not opened until the pilot gate is evaluated. The actor/code/protocol hash is unchanged across stages.

## Metrics fixed before evaluation

Primary: equally weighted mean signal-window integrated squared actual jerk. Window x=2200–3300 m, substeps assigned by position midpoint, as in the existing short study. Report the full per-condition values and every incomplete window.

Secondary: full-trip integrated squared jerk and RMS; event/full peak jerk; total time and energy; signal-window time, energy, friction, strong-braking time (a<-2 m/s^2) and minimum acceleration; settled completions, red crossings, local overspeed against curve/road limits, actuator-boundary violations and fallback counts. The strong-braking threshold is descriptive, not a passenger-comfort standard.

Mechanism diagnostics: on each actually visited state, compute both information-conditioned execution outputs for the same command without stepping the alternative controller. Report paired-state differences and their first occurrence; do not call the alternate output a realized counterfactual trajectory. Closed-loop actor commands may subsequently differ because their inputs follow different states. Confirm the trajectories agree up to their first executed-action difference. Report checks for countdown access, effective use and actual intervention separately.

## Gates and bounded computing

Audit before smoke: identical actor/state-to-command output at identical observation; dependency/weight hashes; no countdown access in color arm or outside 1000 m; red/unknown/out-of-range execution equivalence; a late-green example with a meaningful stop-or-clear restriction; exact actual-jerk ledger; local-limit and terminal accounting; deterministic interrupted/resumed rollout equivalence.

Smoke passes on valid finite records, both settled completions without red/actuator violations. Scientific success is not required for this code gate. Pilot stops on a numerical/ledger error; scientific failures are retained and all nine conditions are completed if the numerical implementation is sound.

Reporting expansion requires all of: (1) both arms complete 9/9 without red, actuator or local-overspeed violations; (2) at least one paired-state execution difference >1e-6 m/s^2; (3) mean signal-window squared-jerk integral decreases by >1e-6; (4) full-trip mean squared-jerk integral does not increase by more than 1e-6; (5) mean travel time increases <=5% and mean energy increases <=3%; (6) peak jerk does not worsen by more than 0.1 m/s^3 and total fallback count does not increase. These are development expansion criteria, not statistical significance or proof of generalization. If no effect or this gate fails, stop without tuning or opening reporting conditions; report the reason.

Before reporting, record measured seconds/trip, total candidate runtime and compute charge. No paid service or GPU is used; external compute charge is zero. Wall-clock stages are upper bounds, not mandatory idle durations. Save complete environment/command/observation progress at actor boundaries and completed-condition records atomically; guard resumption with source and weight hashes. No training state exists to resume because there is no training.

## Reporting boundary

This experiment cannot validate visual perception, a ToMe encoder, the full value of SPaT, a new general control principle, subjective comfort, long-route stability, or the old eight-seed results under a different deadline. An informative zero or adverse effect is retained; it must not be replaced by another actor, range, phase grid, or metric after inspection.
