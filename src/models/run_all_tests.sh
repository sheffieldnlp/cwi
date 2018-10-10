#!/bin/bash
# Run all the experiments

# Monolingual Experiments
python src/models/run_monolingual.py -l english
python src/models/run_monolingual.py -l spanish
python src/models/run_monolingual.py -l german
python src/models/run_monolingual.py -l french

# Crosslingual Experiments
python src/models/run_crosslingual.py -l english
python src/models/run_crosslingual.py -l spanish
python src/models/run_crosslingual.py -l german
python src/models/run_crosslingual.py -l french
