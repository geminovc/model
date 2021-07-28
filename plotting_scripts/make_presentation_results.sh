name1=original

# experiment_path_1=/data/pantea/pantea_experiments_chunky/per_person/from_paper/runs/more_yaws_original_easy_diff_combo
experiment_path_1=/data/pantea/pantea_experiments_chunky/per_person/from_paper/runs/old/close_source_target_original_easy_diff_combo

name2=G_tex_frozen

# experiment_path_2=/video-conf/scratch/pantea_experiments_mapmaker/per_person/from_paper/runs/more_yaws_uniform_easy_diff_combo
experiment_path_2=/data/pantea/pantea_experiments_chunky/per_person/from_paper/runs/original_frozen_Gtex_from_identical

name3=baseline
experiment_path_3=/video-conf/scratch/pantea_experiments_mapmaker/per_person/from_paper/runs/baseline_voxceleb2_easy_diff_combo
# experiment_path_3=/video-conf/scratch/pantea_experiments_mapmaker/per_person/from_paper/runs/more_yaws_baseline_voxceleb2_easy_diff_combo


result_name=G_tex_frozen
# result_name=less_yaws_orig_vs_uni_vs_baseline
# result_name=more_yaws_orig_vs_uni_vs_baseline
# result_name=last_week_difficult_easy_combo


for phase in test unseen_test
do

combo_path_1=${experiment_path_1}/metrics_${phase}_combined_pose.pkl
easy_path_1=${experiment_path_1}/metrics_${phase}_easy_pose.pkl
hard_path_1=${experiment_path_1}/metrics_${phase}_hard_pose.pkl

combo_path_2=${experiment_path_2}/metrics_${phase}_combined_pose.pkl
easy_path_2=${experiment_path_2}/metrics_${phase}_easy_pose.pkl
hard_path_2=${experiment_path_2}/metrics_${phase}_hard_pose.pkl

combo_path_3=${experiment_path_3}/metrics_${phase}_combined_pose.pkl
easy_path_3=${experiment_path_3}/metrics_${phase}_easy_pose.pkl
hard_path_3=${experiment_path_3}/metrics_${phase}_hard_pose.pkl

# if [[ "$phase" == "unseen_test" ]]
# then
#     window=1
# else 
#     window=1
# fi 

window=3

echo "${phase} and ${window}"

python summarize_reconstuction_exps.py \
--result-file-list ${combo_path_1} ${easy_path_1} ${hard_path_1} ${combo_path_2} ${easy_path_2} ${hard_path_2} ${combo_path_3} ${easy_path_3} ${hard_path_3} \
--experiment-name-list ${name1} ${name1} ${name1} ${name2} ${name2} ${name2} ${name3} ${name3} ${name3} \
--pose-name-list combo easy hard combo easy hard combo easy hard \
--window ${window} \
--result-file-name ${phase}_${result_name}.csv

# ./metrics_group_barchart.R ${phase}_${result_name}.csv ./graphs/${result_name}/${phase}

echo "${phase}_${result_name} done!"

done

phase=train

combo_path_1=${experiment_path_1}/metrics_${phase}.pkl
combo_path_2=${experiment_path_2}/metrics_${phase}.pkl
combo_path_3=${experiment_path_3}/metrics_${phase}.pkl
window=1

python summarize_reconstuction_exps.py \
--result-file-list ${combo_path_1} ${combo_path_2} ${combo_path_3} \
--experiment-name-list ${name1} ${name2} ${name3} \
--skip-pose-distribution-data \
--window ${window} \
--result-file-name ${phase}_${result_name}.csv


# ./metrics_bar_plot.R ${phase}_${result_name}.csv ./graphs/${result_name}/${phase}

echo "${phase}_${result_name} done!"