# training from released checkpoints
./train_from_paper_checkpoints.sh  "chunky" "No_frozen_with_metrics" "per_person" 7000 10 5 False False

# training from base
./train_from_base.sh  "chunky" "base_with_metrics" "per_person" 7000 10 5 False False

# debugging
./train_from_paper_checkpoints.sh  "chunky" "debug" "per_person" 7000 1 1 False False