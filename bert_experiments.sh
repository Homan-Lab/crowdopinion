#!/bin/bash

declare dataset_id=$1
declare dataset=$2
declare date=$3 

declare measure=all
declare machine_ID=RIT_PC
declare min_epsilon=0.0
declare max_epsilon=10
declare votes=500
declare Iterations=100
declare samples=1000
declare lower=2	#clustering limits
declare upper=5 #clustering limits
declare majority_flag=True #for baseline calculation
declare Iterations=50


classification_experiments(){
	local EXP=$1
	local results_db=$2
	local weight=$3
	for value in {0..10}
	do
		echo "Experiments on "$dataset
		python CNN_train_vects.py --input_train_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_train.json --input_train_file_vects data/"$dataset_id"/bert_embeddings/X_train.npy  --input_dev_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_dev.json --input_dev_file_vects data/"$dataset_id"/bert_embeddings/X_dev.npy  --output_model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn/"$dataset"_split_cnn.pkl --folder_name data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn
		python CNN_predict.py --input_test_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_test.json --input_test_file_vects data/"$dataset_id"/bert_embeddings/X_test.npy --input_model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn/"$dataset"_split_cnn.pkl --output_pred_name data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn_predict/"$dataset"_predict_test.json
		python model_evaluation.py --model_type CNN --input_test_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_test.json --input_model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn/"$dataset"_split_cnn.pkl --input_pred_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn_predict/"$dataset"_predict_test.json --json_log_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn_predict/"$dataset"_log.json --run_location "$machine_ID" --process_id "$dataset"_"$EXP"_CNN --db_name "$results_db"
		python model_evaluation.py --model_type CNN --input_test_file data/"$dataset_id"/processed/"$dataset"_test.json --input_model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn/"$dataset"_split_cnn.pkl --input_pred_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn_predict/"$dataset"_predict_test.json --json_log_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn_predict/"$dataset"_log.json --run_location "$machine_ID" --process_id "$dataset"_"$EXP"_EMP_CNN --db_name "$results_db" --weight "$weight"
		
		# Training set Experiments for CNN
		# python CNN_predict.py --input_test_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_train.json --input_test_file_vects data/"$dataset_id"/bert_embeddings/X_train.npy --input_model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn/"$dataset"_split_cnn.pkl --output_pred_name data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn_predict/"$dataset"_predict_train.json
		# python model_evaluation.py --model_type CNN --input_test_file data/"$dataset_id"/processed/"$dataset"_train.json --input_model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn/"$dataset"_split_cnn.pkl --input_pred_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn_predict/"$dataset"_predict_train.json --json_log_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn_predict/"$dataset"_log.json --run_location "$machine_ID" --process_id "$dataset"_"$EXP"_EMP_CNN --db_name "$results_db"

		# echo "Experiments on "$dataset
		# python LSTM_train.py --input_train_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_train.json --input_dev_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_dev.json --input_test_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_test.json --output_model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/lstm_predict/"$dataset"_split_lstm.pkl --folder_name data/"$dataset_id"/processed/"$dataset"/"$EXP"/lstm_predict --output_pred_name data/"$dataset_id"/processed/"$dataset"/"$EXP"/lstm_predict/"$dataset"_predict_test.json
		# python model_evaluation.py --model_type LSTM --input_test_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_test.json --input_model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/lstm_predict/"$dataset"_split_lstm.pkl --input_pred_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/lstm_predict/"$dataset"_predict_test.json --json_log_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/lstm_predict/"$dataset"_log.json --run_location "$machine_ID" --process_id "$dataset"_"$EXP"_LSTM --db_name "$results_db"
		# python model_evaluation.py --model_type LSTM --input_test_file data/"$dataset_id"/processed/"$dataset"_test.json --input_model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/lstm_predict/"$dataset"_split_lstm.pkl --input_pred_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/lstm_predict/"$dataset"_predict_test.json --json_log_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/lstm_predict/"$dataset"_log.json --run_location "$machine_ID" --process_id "$dataset"_"$EXP"_EMP_LSTM --db_name "$results_db"
		# python CNN_train_vects.py --input_train_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_train.json --input_train_file_vects data/"$dataset_id"/bert_embeddings/X_train.npy  --input_dev_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_dev.json --input_dev_file_vects data/"$dataset_id"/bert_embeddings/X_dev.npy  --output_model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn/"$dataset"_split_cnn.pkl --folder_name data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn

		# Max Ent Classifier
		# python maxent_model_classifier.py --input_train_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_train.json --input_train_file_vects data/"$dataset_id"/bert_embeddings/X_train.npy --input_dev_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_dev.json --input_dev_file_vects data/"$dataset_id"/bert_embeddings/X_dev.npy --output_model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/maxent/"$dataset".pkl --folder_name data/"$dataset_id"/processed/"$dataset"/"$EXP"/maxent
		# python CNN_predict.py --input_test_file /home/cyril/DataDrive/Experiments/pldl/modeling_annotators/datasets/sbic_feb7/pldl/sbic_intent_test.json --input_test_file_vects /home/cyril/DataDrive/Experiments/pldl/modeling_annotators/datasets/sbic_feb7/modeling_annotator_nn/X_test.npy --input_model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/maxent/"$dataset".pkl --output_pred_name data/"$dataset_id"/processed/"$dataset"/"$EXP"/maxent_predict/"$dataset"_predict_test.json
		
		# python CNN_predict.py --input_test_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_test.json --input_test_file_vects data/"$dataset_id"/bert_embeddings/X_test.npy --input_model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/maxent/"$dataset".pkl --output_pred_name data/"$dataset_id"/processed/"$dataset"/"$EXP"/maxent_predict/"$dataset"_predict_test.json
		# python model_evaluation.py --model_type CNN --input_test_file data/"$dataset_id"/processed/"$dataset"_test.json --input_model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn/"$dataset"_split_cnn.pkl --input_pred_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn_predict/"$dataset"_predict_test.json --json_log_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/cnn_predict/"$dataset"_log.json --run_location "$machine_ID" --process_id "$dataset"_"$EXP"_EMP_CNN --db_name "$results_db"

		# python model_evaluation.py --model_type MaxEnt --input_test_file data/"$dataset_id"/processed/"$dataset"_test.json --input_model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/maxent/"$dataset".pkl --input_pred_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/maxent_predict/"$dataset"_predict_test.json --json_log_file results/"$dataset_id"/"$dataset"/"$EXP"/maxent/"$dataset"_log.csv --run_location "$machine_ID" --process_id "$dataset"_"$EXP"_EMP_MAXENT --db_name  "$results_db" --weight "$weight"
					
	done
}

pooling_experiments()
{
	#pooling experiments implmented with sklearn.
	local cluster_name=$1
	local EXP=$2
	local results_db=$3
	local db=$4
	local hybrid=$5

	python "$cluster_name"_train.py --train_file data/"$dataset_id"/processed/"$dataset"_train.json  --train_file_vects data/"$dataset_id"/bert_embeddings/X_train.npy --dev_file data/"$dataset_id"/processed/"$dataset"_dev.json --dev_file_vects data/"$dataset_id"/bert_embeddings/X_dev.npy --lower "$lower" --upper "$upper" --iterations 50 --output_file "$dataset"_split_"$cluster_name"  --folder_name data/"$dataset_id"/processed/"$dataset"/"$EXP" --nlp_data True --glove bert --hybrid "$hybrid"
	upper=$((upper -1))
	for topic in $(seq $lower $upper)
	do
		echo "Sampling on "$cluster_name" "$topic
		python "$cluster_name"_predict.py --file_to_predict data/"$dataset_id"/processed/"$dataset"_train.json --file_to_predict_vects data/"$dataset_id"/bert_embeddings/X_train.npy --model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/logs/models/CL"$topic".pkl --output_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/logs/"$dataset"_predict_train.json --nlp_data True --cluster_info data/"$dataset_id"/processed/"$dataset"/"$EXP"/logs/models/cluster_info_"$topic".json  --embeddings bert --hybrid "$hybrid"

		python model_selection_pooling.py --model_type "$EXP" --sampler cluster --topics "$topic" --votes "$votes" --input_test_file data/"$dataset_id"/processed/"$dataset"_train.json --n_samples_to_draw 100 --n_iterations "$Iterations" --input_pred_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/logs/"$dataset"_predict_train.json --json_log_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/logs/"$dataset"_log_"$EXP"_only.json --run_location "$machine_ID" --process_id "$dataset"_"$cluster_name"_"$measure" --db_name "$db" 
	done

	python sampling_graphs.py --db_name "$db" --col_name "$dataset"_"$cluster_name"_"$measure"_cluster --path_to_save results/"$date"/"$EXP" --db_to_save "$results_db" --col_to_save "$dataset"_"$EXP"_model --json_log_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/logs/"$dataset"_log_"$EXP"_only.json --model_selection data/"$dataset_id"/processed/"$dataset"/"$EXP"/logs/"$dataset"_"$EXP"_model.json

	for split in train dev test
	do
		python "$cluster_name"_predict.py --file_to_predict data/"$dataset_id"/processed/"$dataset"_"$split".json --file_to_predict_vects data/"$dataset_id"/bert_embeddings/X_"$split".npy --model_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/logs/models/CLX.pkl --result_db_name data/"$dataset_id"/processed/"$dataset"/"$EXP"/logs/"$dataset"_"$EXP"_model.json --result_exp_name "$dataset"_"$EXP"_model --output_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_"$split".json --nlp_data True --cluster_info data/"$dataset_id"/processed/"$dataset"/"$EXP"/logs/models/cluster_info_X.json --embeddings bert --hybrid "$hybrid"
	done

	python model_evaluation.py --model_type "$cluster_name" --input_test_file data/"$dataset_id"/processed/"$dataset"_test.json --input_model_file None --input_pred_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_test.json --json_log_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$cluster_name"_eval.json --run_location "$machine_ID" --process_id "$dataset"_"$cluster_name"_predict_"$measure" --db_name "$results_db" --weight "$hybrid"
		
}

for hybrid in False 25 50 75 100 #w parameter 50 75 25
do
	declare results_db="$date"_HY"$hybrid""$dataset_id"Results
	declare db="$date"_"$dataset_id"HY"$hybrid"
	for Pooling in lda gmm kmeans NBP FMM None MaxEnt
	do

		echo $Pooling
		if [ "$hybrid" == "False" ]
		then
		declare EXP="$Pooling"_predict
		else
		declare EXP="$Pooling"HY"$hybrid"_predict
		fi
		rm data/"$dataset"/"$EXP"/"$dataset"_log_"$EXP"_only.json
		mkdir data/"$dataset"/"$EXP"/
		if [ "$Pooling" == "NBP" ]; then

			python prep_experiments.py --folder_name data/"$dataset_id"/processed/"$dataset"/"$EXP"

			for epsilon in $(seq 0.1 0.5 15)
			do
				echo "Experiments on "$dataset
				echo "Epsilon "$epsilon
					python k_neighborhood_train.py --train_file data/"$dataset_id"/processed/"$dataset"_train.json --dev_file data/"$dataset_id"/processed/"$dataset"_dev.json --test_file data/"$dataset_id"/processed/"$dataset"_test.json --max_epsilon "$epsilon" --output "$dataset" --measure "$measure" --folder_name data/"$dataset_id"/processed/"$dataset"/"$EXP" --measure KL --nlp_data True --glove True --hybrid "$hybrid"
					python model_selection_pooling.py --model_type "$EXP" --sampler "$EXP" --topics "$epsilon" --votes "$votes" --input_test_file data/"$dataset_id"/processed/"$dataset"_train.json --n_samples_to_draw "$samples" --n_iterations "$Iterations" --input_pred_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_train.json --json_log_file data/"$dataset"/"$EXP"/"$dataset"_log_"$EXP"_only.json --run_location "$machine_ID" --process_id "$dataset"_NBP_"$measure" --db_name "$db"
			done

			python sampling_graphs.py --db_name "$db" --col_name "$dataset"_NBP_"$measure"_"$EXP" --path_to_save results/"$date"/"$EXP" --db_to_save "$results_db" --col_to_save "$dataset"_"$EXP"_model --json_log_file data/"$dataset"/"$EXP"/"$dataset"_log_"$EXP"_only.json --model_selection data/"$dataset_id"/processed/"$dataset"/"$EXP"/logs/"$dataset"_"$EXP"_model.json
			python k_neighborhood_train.py --train_file data/"$dataset_id"/processed/"$dataset"_train.json --dev_file data/"$dataset_id"/processed/"$dataset"_dev.json --test_file data/"$dataset_id"/processed/"$dataset"_test.json --output "$dataset" --measure "$measure" --folder_name data/"$dataset_id"/processed/"$dataset"/"$EXP" --measure KL --nlp_data True --results_db "$results_db" --results_col "$dataset"_"$EXP"_model --glove True
			python model_evaluation.py --model_type NBP --input_test_file data/"$dataset_id"/processed/"$dataset"_test.json --input_model_file None --input_pred_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_test.json --json_log_file data/"$dataset_id"/processed/"$dataset"/nbp_predict/nbp_eval.json --run_location "$machine_ID" --process_id "$dataset"_NBP_predict_"$measure" --db_name "$results_db" --weight "$hybrid"
			
		fi	
		
		if [ "$Pooling" == "gmm" ] || [ "$Pooling" == "kmeans" ] || [ "$Pooling" == "lda" ]; then
			echo "Experiments on "$dataset
	
			python prep_experiments.py --folder_name data/"$dataset_id"/processed/"$dataset"/"$EXP"
			pooling_experiments $Pooling $EXP $results_db $db $hybrid

		fi

		if [ "$Pooling" == "FMM" ]; then
			mkdir data/"$dataset_id"/processed/"$dataset"/"$EXP"/sample_test
			python fmm_train.py --input_file data/"$dataset_id"/processed/"$dataset"_train.json --dev_file data/"$dataset_id"/processed/"$dataset"_dev.json --test_file data/"$dataset_id"/processed/"$dataset"_test.json --input_train_file_vectors data/"$dataset_id"/bert_embeddings/X_train.npy --dev_file_vectors data/"$dataset_id"/bert_embeddings/X_dev.npy --test_file_vectors data/"$dataset_id"/bert_embeddings/X_test.npy --nlp_flag vectors --lower "$lower" --upper "$upper" --output_file "$dataset"_predict --folder_name data/"$dataset_id"/processed/"$dataset"/"$EXP"/sample_test --hybrid "$hybrid" 
			for topic in $(seq $lower $upper)
			do
				echo "Experiments on "$dataset
				python model_selection_pooling.py --model_type "$EXP" --sampler cluster --topics "$topic" --votes "$votes" --input_test_file data/"$dataset_id"/processed/"$dataset"_train.json --n_samples_to_draw 100 --n_iterations "$Iterations" --input_pred_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/sample_test/CL"$topic"/"$dataset"_predict_train.json --json_log_file data/"$dataset"/"$EXP"/"$dataset"_log_"$EXP"_only.json --run_location "$machine_ID" --process_id "$dataset"_fmm_"$measure" --db_name "$db"
			done
			python sampling_graphs.py --db_name "$db" --col_name "$dataset"_fmm_"$measure"_cluster --path_to_save results/"$date"/"$EXP" --db_to_save "$results_db" --col_to_save "$dataset"_"$EXP"_model --json_log_file data/"$dataset"/"$EXP"/"$dataset"_log_"$EXP"_only.json --model_selection data/"$dataset_id"/processed/"$dataset"/"$EXP"/logs/"$dataset"_"$EXP"_model.json
			python fmm_predict.py --input_folder data/"$dataset_id"/processed/"$dataset"/"$EXP"/sample_test/CLX --result_db_name "$results_db" --result_exp_name "$dataset"_"$EXP"_model --output_folder data/"$dataset_id"/processed/"$dataset"/"$EXP" --output_process_id "$dataset"_predict
			python model_evaluation.py --model_type FMM --input_test_file data/"$dataset_id"/processed/"$dataset"_train.json --input_pred_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/"$dataset"_predict_train.json --json_log_file data/"$dataset_id"/processed/"$dataset"/"$EXP"/fmm_eval.json --run_location "$machine_ID" --process_id "$dataset"_FMM_predict_"$measure" --db_name "$results_db" --weight "$hybrid"
		fi


		if [ "$Pooling" == "MaxEnt" ]; then #baseline
			for value in {0..10}
				do
					echo "Experiments on "$dataset
					python maxent_model_classifier.py --input_train_file data/"$dataset_id"/processed/"$dataset"_train.json --input_train_file_vects data/"$dataset_id"/bert_embeddings/X_train.npy --input_dev_file data/"$dataset_id"/processed/"$dataset"_dev.json --input_dev_file_vects data/"$dataset_id"/bert_embeddings/X_dev.npy --output_model_file data/"$dataset_id"/processed/"$dataset"/maxent/"$dataset".pkl --folder_name data/"$dataset_id"/processed/"$dataset"/maxent
					
					python CNN_predict.py --input_test_file data/"$dataset_id"/processed/"$dataset"_test.json --input_test_file_vects data/"$dataset_id"/bert_embeddings/X_test.npy --input_model_file data/"$dataset_id"/processed/"$dataset"/maxent/"$dataset".pkl --output_pred_name data/"$dataset_id"/processed/"$dataset"/maxent/"$dataset"_predict_test.json
					python model_evaluation.py --model_type "$Pooling" --input_test_file data/"$dataset_id"/processed/"$dataset"_test.json --input_model_file data/"$dataset_id"/processed/"$dataset"/maxent/"$dataset".pkl --input_pred_file data/"$dataset_id"/processed/"$dataset"/maxent/"$dataset"_predict_test.json --json_log_file results/"$dataset_id"/"$dataset"/maxent/"$dataset"_log.csv --run_location "$machine_ID" --process_id "$dataset"_maxent --db_name  "$results_db"
					
				done
		fi

		if [ "$Pooling" == "None" ]; then #baseline
			for value in {0..10}
				do
					# echo "Experiments on "$dataset
					# python LSTM_train.py --input_train_file data/"$dataset_id"/processed/"$dataset"_train.json --input_dev_file data/"$dataset_id"/processed/"$dataset"_dev.json --input_test_file data/"$dataset_id"/processed/"$dataset"_test.json --output_model_file data/"$dataset_id"/processed/"$dataset"/lstm_only_predict/"$dataset"_split_lstm.pkl --folder_name data/"$dataset_id"/processed/"$dataset"/lstm_only_predict --output_pred_name data/"$dataset_id"/processed/"$dataset"/lstm_only_predict/"$dataset"_predict_test.json --majority "$majority_flag"
					# python model_evaluation.py --model_type LSTM --input_test_file data/"$dataset_id"/processed/"$dataset"_test.json --input_model_file data/"$dataset_id"/processed/"$dataset"/lstm_only_predict/"$dataset"_split_lstm.pkl --input_pred_file data/"$dataset_id"/processed/"$dataset"/lstm_only_predict/"$dataset"_predict_test.json --json_log_file data/"$dataset_id"/processed/"$dataset"/lstm_only_predict/"$dataset"_log.json --run_location "$machine_ID" --process_id "$dataset"_LSTM --db_name  "$results_db"

					# echo "Experiments on "$dataset

					python CNN_train_vects.py --input_train_file data/"$dataset_id"/processed/"$dataset"_train.json --input_train_file_vects data/"$dataset_id"/bert_embeddings/X_train.npy --input_dev_file data/"$dataset_id"/processed/"$dataset"_dev.json --input_dev_file_vects data/"$dataset_id"/bert_embeddings/X_dev.npy --output_model_file data/"$dataset_id"/processed/"$dataset"/cnn_only/"$dataset"_split_cnn.pkl --folder_name data/"$dataset_id"/processed/"$dataset"/cnn_only
					python CNN_predict.py --input_test_file data/"$dataset_id"/processed/"$dataset"_test.json --input_test_file_vects data/"$dataset_id"/bert_embeddings/X_test.npy --input_model_file data/"$dataset_id"/processed/"$dataset"/cnn_only/"$dataset"_split_cnn.pkl --output_pred_name data/"$dataset_id"/processed/"$dataset"/cnn_only/"$dataset"_predict_test.json
					python model_evaluation.py --model_type CNN --input_test_file data/"$dataset_id"/processed/"$dataset"_test.json --input_model_file data/"$dataset_id"/processed/"$dataset"/cnn_only/"$dataset"_split_cnn.pkl --input_pred_file data/"$dataset_id"/processed/"$dataset"/cnn_only/"$dataset"_predict_test.json --json_log_file data/"$dataset_id"/processed/"$dataset"/cnn_only/"$dataset"_log.csv --run_location "$machine_ID" --process_id "$dataset"_CNN --db_name  "$results_db"
					
					# python CNN_predict.py --input_test_file data/"$dataset_id"/processed/"$dataset"_train.json --input_test_file_vects data/"$dataset_id"/bert_embeddings/X_train.npy --input_model_file data/"$dataset_id"/processed/"$dataset"/cnn_only/"$dataset"_split_cnn.pkl --output_pred_name data/"$dataset_id"/processed/"$dataset"/cnn_only/"$dataset"_predict_train.json
					# python model_evaluation.py --model_type CNN --input_test_file data/"$dataset_id"/processed/"$dataset"_train.json --input_model_file data/"$dataset_id"/processed/"$dataset"/cnn_only/"$dataset"_split_cnn.pkl --input_pred_file data/"$dataset_id"/processed/"$dataset"/cnn_only/"$dataset"_predict_train.json --json_log_file data/"$dataset_id"/processed/"$dataset"/cnn_only/"$dataset"_log.json --run_location "$machine_ID" --process_id "$dataset"_CNN --db_name  "$results_db"

				done
		fi

		if [ "$Pooling" == "gmm" ] || [ "$Pooling" == "kmeans" ] || [ "$Pooling" == "lda" ] || [ "$Pooling" == "NBP" ] || [ "$Pooling" == "FMM" ]; then
			classification_experiments $EXP $results_db $hybrid
		fi


	done
done
