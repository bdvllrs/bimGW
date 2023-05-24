# 0.2.0

* The release version is added under the `_code_version` configuration value
  when loading the config with `get_args`.
* Remove: wrong assert in shapes dataset.
* Remove: logging of removed cosine_coef.
* Update: out-of-distribution training to work with the DomainItems
  inputs.
* Fix issues when loading pretrained global workspace from csv in
  out-of-distribution script.
* Add type annotations to `get_args`.
* Fix: use seed information when filtering the correct pretrained global
  workspace in the odd-out-out experiment when
  using `select_row_from_current_coefficients`.
* Change: `filter_sync_domain` so that experiments with lower
  available_sync_domains still work properly
* Add config values for better visualization
* Reformat using black

# 0.1.0

Initial version.
