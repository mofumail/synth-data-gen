Requires uv  
Run uv sync once after cloning, then uv run python main.py

Raw Synerise parquets expected at ../DATA/ relative to code/pipeline/                    
  (edit DATA_DIR in config.py to change).

Pipeline (run from code/pipeline/):                                                      
    uv run python preprocess.py
    uv run python ingestion/svdpq.py                                                       
    uv run python train.py                                                                 
    uv run python evaluate.py
        --n-sessions <int>   max synthetic sessions per seed              
        --num-seeds  <int>   number of seeds   
        and other flags, but upper 2 are most important


    An easier way to run it is with :
    uv run python main.py --from "stage" (train, evaluate etc)" with whatever relevant flag you'd want to pass through (--n-sessions, --num-seeds) This runs the entire pipeline end-to-end without having to manually start the next stage.

Output folder names are based on hyperparams & datetime when running the script. Add run flag --comment to add an extra bit of text to the output folder name.