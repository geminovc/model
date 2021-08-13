# This script reads the experimets numerical results in the paths and make the bar graphs  
# Sample uusage:
# bash make_presentation_results.sh name1 path1 name2 path2 ... namei pathi result_name
 
experiment_names=()
experiment_paths=()

for (( index_odd=1; index_odd< $# ; index_odd+=2 )); do
        index_even=$((index_odd+1))
        experiment_names+=(${!index_odd})
        experiment_paths+=(${!index_even})
done

result_name=${!#}

for phase in train test unseen_test
do
        window=3
        args="--window ${window} --result-file-name ${phase}_${result_name}.csv"
        path_lists="--result-file-list "
        name_lists="--experiment-name-list "
        pose_lists="--pose-name-list"

        for (( index=0; index< ${#experiment_names[@]} ; index+=1 )); do

                if [[  "$phase" == "train" ]]; then
                        path_lists="${path_lists} ${experiment_paths[$index]}/metrics_${phase}.pkl" 
                        name_lists="${name_lists} ${experiment_names[$index]}"
                else
                        path_lists="${path_lists} ${experiment_paths[$index]}/metrics_${phase}_combined_pose.pkl ${experiment_paths[$index]}/metrics_${phase}_easy_pose.pkl ${experiment_paths[$index]}/metrics_${phase}_hard_pose.pkl" 
                        name_lists="${name_lists} ${experiment_names[$index]} ${experiment_names[$index]} ${experiment_names[$index]}"
                        pose_lists="${pose_lists} combo easy hard"
                fi

        done

        args="${args} ${path_lists} ${name_lists} ${pose_lists}"

        if [[ "$phase" == "train" ]]; then
                args="${args} --skip-pose-distribution-data "
        fi

        echo "$phase"
        echo " ${args}"

        python summarize_reconstuction_exps.py ${args}

        # if [[ "$phase" == "train" ]]; then
                # ./metrics_bar_plot.R ${phase}_${result_name}.csv ./graphs/${result_name}/${phase}
        # else
                # ./metrics_group_barchart.R ${phase}_${result_name}.csv ./graphs/${result_name}/${phase}
        # fi

        echo "${phase}_${result_name} done!"
done
