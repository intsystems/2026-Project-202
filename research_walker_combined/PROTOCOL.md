# Combined phase tracking and temporal policy regularization

New exploratory follow-up; earlier results remain unchanged. Frozen before new
training and MG. Motivation: tracking and smoothness individually did not yield
consistent whole-state regularity. Test their combination, not a new sensor
chosen for a favorable MG. Post-hoc old sensor/action-log analyses are separate.

Task, physical simulator, reference, known phase, reset distribution, frozen
normalization and shared initial policy are copied from research_walker_phase_wide.
This is local phase-conditioned gait maintenance, NOT autonomous default-reset
Walker2d and NOT the dynamics of neural-network training weights.

Pilot seed280 is a 2x2 factorial: tracking coefficient0/3 crossed with temporal
policy regularization coefficient0/1. No coefficient sweep or replacement.
Tracking modifies original reward with the previous broad width38.477382693753626.
Smoothness adds mean squared differences of clipped deterministic policy means
at consecutive nonterminal rollout observations to PPO loss; gradients through
both means. This implementation is copied from research_walker_smooth, previously
used on a different task; coefficient1 is fixed from that work. Do NOT choose
coefficient from deterministic evaluation J1, since PPO loss has different units.
No filter or action interpolation is applied during testing. Ordinary state
transitions, original health termination and action bounds remain.

Each run1048576 transitions;8env,512steps/env,batch256,5epochs,lr3e-5->3e-6,
clip.1,targetKL.01,gamma.99,GAE.95,entropy0,vf.5,gradclip.5;64x64tanh actor/critic.
Same phase anchor with reset Adam in all arms. Check smooth0 matches ordinary PPO
bit-for-bit8192 transitions and nonzero smooth penalty affects weights.

Validation77001..77005 at the last3 snapshots; gate on final only. Test78001..78010
is disjoint and used after gate frozen.512burn+4096measurement, deterministic
policies, no TimeLimit, actual falls retained and original return zero-padded.
Eligibility and R,D_strobe inherited. C_cycle now PRIMARY independent whole-state
diagnostic (predefined here, previously post-hoc): interpolate each complete cycle
to64 phases; between-cycle variance averaged across phases / total state variance.
J1/J2 and action spectral power/entropy are directly related to the intervention,
not independent evidence that whole-state dimension decreased.

Pilot expansion gate: combined and vanilla at least4 common eligible cycles,
combined vs vanilla median R AND C_cycle<=.8; original all-reset reward>=.9;
at least4 healthy walks and at most1 fewer than each comparator. Additionally
combined vs tracking-alone C_cycle<=.9,R<=1.1,J1<=.8,return>=.9 with>=4common.
Baseline median R>=.02,C_cycle>=.01 required versus vanilla. No MG in gate.
If passed, run ALL five pairs281..285 tracking-alone vs combined, no replacements.
Else stop expansion and test all four pilot arms, report failure. In either case
test all pilot arms, keep pilot descriptive. No further hyperparameter changes.
Strong confirmatory event: R,C_cycle ratios<=.75 vs tracking,>=8common, baseline
R>=.02,C>=.01,return>=.9,at most1 lost walk. Other metrics/failed episodes remain.

MG primary rightknee,W2048,tau8,E20,k20,Theiler312,threewindows. E40 diagnostic.
Secondary leftknee,tau4/16,known14cycle window; all fixed and retained. No
selection based on MG. Compare independent metrics on the same common resets.
Report nondegeneracy and embedding sensitivity, not an exact dimension claim.
Perturbations: first common reset,68signed17Dprobes at0/1024,600steps,epsilon.001,
identical external phase and exact zero replay. Include falls. Finite-time
sensitivity, not Lyapunov/Floquet spectrum. Serial timing only after other jobs.

Deliver concise Russian PDF/Markdown, raw CSV/NPZ, source and audited archive.
Keep every pilot outcome. No manuscript changes. Reset seeds are repeated tests
of a policy; independent training repetitions are counted separately.
