"""
M8 Stage A - SEMI E10 tool-state layer and RAM metrics.

Turns the tool-level event log plus injection metadata into a per-tool E10
state timeline and standard reliability/availability/maintainability metrics:

  - ``e10_states.py``  : ``build_tool_state_timeline`` - tidy per-tool state
                         timeline (PRODUCTIVE, STANDBY, SCHEDULED DOWNTIME,
                         UNSCHEDULED DOWNTIME; ENGINEERING declared but always
                         empty). Exact partition of [0, horizon] per tool.
  - ``ram_metrics.py``  : ``ram_metrics_by_tool`` / ``ram_metrics_by_station`` /
                         ``state_time_decomposition`` - MTBF, MTTR,
                         availability, equipment utilization.
  - ``equipment_check.py`` : Stage A regression + sanity gates (script).

Import convention (matches the rest of ``src/``): modules use BARE imports
and consumers put ``src/equipment`` (plus ``src/generator`` etc. as needed)
on ``sys.path`` - see ``equipment_check.py`` for the pattern.

See each module's docstring for the modeling details and honest-scope
disclaimers (stylized SEMI E10 mapping, not a standards-compliant
implementation).
"""
