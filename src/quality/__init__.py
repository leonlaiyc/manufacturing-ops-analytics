"""
M7 Stage B — lot-level yield ground truth.

Turns the Stage-A event log into a measurable yield target for later M7
stages (virtual metrology, yield-aware what-if):

  - ``queue_time.py``  : post-LITHO queue-time ("photoresist aging" analogy,
                         stylized) windows and violation flags.
  - ``yield_model.py``  : additive, interpretable latent defect-probability
                         model; realized Binomial defect counts and lot
                         yield; noisy virtual-metrology reading.
  - ``quality_check.py`` : regression + calibration gates (run as a script).

See each module's docstring for the modeling details and honest-scope
disclaimers (synthetic, stylized, not a physical or real-fab yield model).
"""
