"""
M7 Stage B/C - lot-level yield ground truth, virtual metrology, chamber matching.

Turns the Stage-A event log into a measurable yield target and downstream
quality analyses:

  - ``queue_time.py``  : post-LITHO queue-time ("photoresist aging" analogy,
                         stylized) windows and violation flags.
  - ``yield_model.py``  : additive, interpretable latent defect-probability
                         model; realized Binomial defect counts and lot
                         yield; noisy virtual-metrology reading.
  - ``quality_check.py`` : Stage B regression + calibration gates (script).
  - ``virtual_metrology.py`` : hand-built OLS predicting metrology_reading
                         from strictly upstream features; time-based split;
                         test R^2 vs train-mean baseline; risk-ranking AUC.
  - ``chamber_matching.py`` : LITHO-1 vs LITHO-2 two-sample tests (t and
                         Mann-Whitney), effect sizes, per-chamber daily-mean
                         series, and a false-positive-guarded mismatch verdict.
  - ``vm_check.py``      : Stage C regression + sanity gates (script).

Import convention (matches the rest of ``src/``): modules use BARE imports
(``from queue_time import ...``) and consumers put ``src/quality`` (plus
``src/generator`` etc. as needed) on ``sys.path`` -- see ``quality_check.py``
or ``vm_check.py`` for the pattern. This file therefore documents the public
names per module rather than importing them (package-relative imports would
break the bare-import convention the modules themselves use).

Public exports by module:

  queue_time         : DEFAULT_WINDOW_HOURS, TARGET_VIOLATION_QUANTILE,
                       post_litho_queue_times, calibrate_window,
                       flag_violations
  yield_model        : QualityConfig, WAFERS_PER_LOT, DEFAULT_METROLOGY_SIGMA,
                       PT_EXCESS_CLIP, etch_pt_excess, build_lot_quality
  virtual_metrology  : FEATURE_COLUMNS, TRAIN_FRACTION, VMResult,
                       build_features, time_split, fit_ols, predict,
                       r2_vs_mean_baseline, risk_ranking_auc, fit_and_evaluate
  chamber_matching   : TwoSampleResult, compare_litho_tools,
                       daily_mean_series, mismatch_verdict

See each module's docstring for the modeling details and honest-scope
disclaimers (synthetic, stylized, not a physical or real-fab yield model).
"""
