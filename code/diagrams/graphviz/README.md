# Pipeline Diagrams

Generated Graphviz diagrams for the synthetic e-commerce session simulation pipeline.

## Files

- 01_c4_system_context: system context for researcher, data sources, artifacts, reports, W&B, and MongoDB.
- 02_c4_container_pipeline: container view of main.py, stage CLIs, packages, config, and artifact stores.
- 03_c4_component_training_and_model: train.py and SessionTransformer component view.
- 04_c4_component_evaluation_and_reporting: evaluate.py, multi-seed orchestration, metrics, baselines, and reports.
- 05_c4_component_simulation_generation: arrivals, identities, generation, validity, batches, uploader, and MongoDB.
- 06_flow_preprocess_svdpq: preprocessing and SVD-PQ data-preparation flow.
- 07_flow_train_py: detailed train.py code flow.
- 08_flow_evaluate_py: detailed evaluate.py code flow, including full and fidelity-only paths.
- 9_flow_autoregressive_generation: identity/history seeding through autoregressive transformer sampling and validity filtering.
- 10_artifacts_data_lineage: lineage map for raw data, caches, model artifacts, synthetic data, GRU4Rec, reports, and MongoDB.

## Notes

- simulation/orchestrator.py references MMPPArrivalModel and TODArrivalModel, the generation diagram marks these as referenced/missing implementations.
- Locked test data is shown separately from train/validation data and marked as final-evaluation-only.
