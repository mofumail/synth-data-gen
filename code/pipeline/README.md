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