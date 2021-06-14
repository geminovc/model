# Since we pretty much only run trials on the per_person dataset we're only going 
# to have those trials here.
# However, its really easy to change the dataset just change the per_person to something else

# The standard no unet trial from base with masks applied
# The no mask trials are not very successful yet, so we haven't included those
./unet_run_script.sh False "no_unet_from_base" "from_base" "per_person" 2 10000 10 10 True

# Now from paper checkpoints
./unet_run_script.sh False "no_unet_from_paper" "from_paper" "per_person" 2 10000 10 10 True

# Now the hf input unet from base
./unet_run_script.sh False "hf_unet_from_base" "from_base" "per_person" 2 10000 10 10 True 16 "hf"

# Hf input unet from paper
./unet_run_script.sh False "hf_unet_from_paper" "from_base" "per_person" 2 10000 10 10 True 16 "hf"

# Now the hf and lf input unet from base
./unet_run_script.sh False "hf_lf_unet_from_base" "from_base" "per_person" 2 10000 10 10 True 16 "hf, lf"

# Hf and lf input unet from paper
./unet_run_script.sh False "hf_lf_unet_from_paper" "from_base" "per_person" 2 10000 10 10 True 16 "hf, lf"
