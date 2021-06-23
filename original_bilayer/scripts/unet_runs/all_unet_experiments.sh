# Since we pretty much only run trials on the per_person dataset we're only going 
# to have those trials here.
# However, its really easy to change the dataset just change the per_person to something else like general or per_vide

# The standard no unet trial from base with masks applied
# The no mask trials are not very successful yet, so we haven't included those
./unet_run_script.sh False "no_unet_from_base_rebalanced" "from_base" "per_person" 2 10000 10 10 True True True

# Now from paper checkpoints
./unet_run_script.sh False "no_unet_from_paper_rebalanced" "from_paper" "per_person" 2 10000 10 10 True True True

# Now the hf input unet from base
./unet_run_script.sh True "hf_unet_from_base_rebalanced" "from_base" "per_person" 2 10000 10 10 True True True "hf" 16 

# Hf input unet from paper
./unet_run_script.sh True "hf_unet_from_paper_rebalanced" "from_paper" "per_person" 2 10000 10 10 True True True "hf" 16 

# Now the hf and lf input unet from base
./unet_run_script.sh True "hf_lf_unet_from_base_rebalanced" "from_base" "per_person" 2 10000 10 10 True True True "hf,\ lf" 19 

# Hf and lf input unet from paper
./unet_run_script.sh True "hf_lf_unet_from_paper_rebalanced" "from_paper" "per_person" 2 10000 10 10 True True True "hf,\ lf" 19 




# These are the no-rebalanced versions of those above experiments
./unet_run_script.sh False "no_unet_from_base" "from_base" "per_person" 2 10000 10 10 True True False

# Now from paper checkpoints
./unet_run_script.sh False "no_unet_from_paper" "from_paper" "per_person" 2 10000 10 10 True True False

# Now the hf input unet from base
./unet_run_script.sh True "hf_unet_from_base" "from_base" "per_person" 2 10000 10 10 True True False "hf" 16 

# Hf input unet from paper
./unet_run_script.sh True "hf_unet_from_paper" "from_paper" "per_person" 2 10000 10 10 True True False "hf" 16 

# Now the hf and lf input unet from base
./unet_run_script.sh True "hf_lf_unet_from_base" "from_base" "per_person" 2 10000 10 10 True True False "hf,\ lf" 19 

# Hf and lf input unet from paper
./unet_run_script.sh True "hf_lf_unet_from_paper" "from_paper" "per_person" 2 10000 10 10 True True False "hf,\ lf" 19 
