Requires uv. 
Run uv sync once after cloning, then uv run python <script>.py

Raw Synerise parquets expected at ../DATA/ relative to code/pipeline/                    
  (edit DATA_DIR in config.py to change).

Pipeline (run from code/pipeline/):                                                      
    uv run python preprocess.py
    uv run python ingestion/svdpq.py                                                       
    uv run python train.py                                                                 
    uv run python evaluate.py
        --n-sessions <int>   max synthetic sessions per seed (default 100000)              
        --num-seeds  <int>   number of seeds (default 5)   
        and other flags, but upper 2 are most important

MODEL_NAME takes t, v as input, so all 3 runs should land in disjoint folders
and not overwrite each other

  - output/models/session_transformer_d128_l4_h4_svdpq_t4v512/                             
  - output/models/session_transformer_d128_l4_h4_svdpq_t8v256/
  - output/models/session_transformer_d128_l4_h4_svdpq_t4v2048/  