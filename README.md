# Code for An In-depth Investigation of User Response Simulation for Conversational Search

## Package requirements (recommended versions).
1. python==3.8
2. `pip install -r src/requirements.txt`
3. Download data
    ```
    cd cosearcher
    scripts/bootstrap.sh
    ```


## How to use
1. Preprocess data and generate train/val/test split with long/short cooperativeness split.

    Run `t5.ipynb` cell [1]
    
3. Train the RoBERTa classifier for answer type prediction.
   
    Run `t5.ipynb` cell [2] and [3]

5. Predict answer type with trained RoBERTa classifier.
    
    Run `t5.ipynb` cell [4]
    
3. Finetune T5 and UnifiedQA (on Qulac for example) 
  
    ```
    cd src
    $ deepspeed t5trainer.py \
      --deepspeed ds_config_zero3.json \
      --model_name_or_path t5-small \
      --do_train \
      --do_eval \
      --do_predict \
      --source_prefix "answer: " \
      --output_dir output/t5-small-qulac/ \
      --per_device_train_batch_size=64 \
      --per_device_eval_batch_size=64 \
      --overwrite_output_dir \
      --predict_with_generate \
      --text_column t5-question \
      --summary_column answer \
      --seed 2023 \
      --num_train_epochs 30 \
      --train_file ../data/processed/qulac_train.csv \
      --validation_file ../data/processed/qulac_dev.csv \
      --test_file ../data/processed/qulac_test.csv 
    ```
    
    ```
    $ deepspeed t5trainer.py \
      --deepspeed ds_config_zero3.json \
      --model_name_or_path allenai/unifiedqa-t5-small \
      --do_train \
      --do_eval \
      --do_predict \
      --source_prefix "" \
      --output_dir output/unifiedqa-small-qulac/ \
      --per_device_train_batch_size=64 \
      --per_device_eval_batch_size=64 \
      --overwrite_output_dir \
      --predict_with_generate \
      --text_column unifiedqa-question \
      --summary_column answer \
      --seed 2023 \
      --num_train_epochs 30 \
      --train_file ../data/processed/qulac_train.csv \
      --validation_file ../data/processed/qulac_dev.csv \
      --test_file ../data/processed/qulac_test.csv 

    ```

3. Run document retrieval experiments. (with post-processed t5 results for example)
    ```
    cd cosearcher
    $ python3 src/main.py --output_file_path ../src/output/t5-small-qulac.csv > ../src/output/t5-small-qulac.json
    ```
    
4. Run evaluations in `view.ipynb`
5. Significance test in `sigtest.py`
    
## Reference

Please cite the following work if you use this code repository in your work: