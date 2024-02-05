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
0. Train the RoBERTa classifier for answer type prediction.
    ```
    $ cd src
    $ python3 roberta.py
    ``` 
    
    
2. Preprocess data and generate train/val/test split with long/short cooperativeness split.
    ```
    $ cd src
    $ python3 gen_qulac_data.py
    $ python3 gen_clariq_data.py
    ```
    
2. Finetune T5 and UnifiedQA (with Qulac split for example) 
  
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
      --train_file qulac_train.csv \
      --validation_file qulac_dev.csv \
      --test_file qulac_dev.csv \
    ```
    
    ```
    $ deepspeed t5trainer.py \
      --deepspeed ds_config_zero3.json \
      --model_name_or_path allenai/unifiedqa-t5-small \
      --do_train \
      --do_eval \
      --do_predict \
      --source_prefix "" \
      --output_dir output/unifiedqa-small-clariq/ \
      --per_device_train_batch_size=64 \
      --per_device_eval_batch_size=64 \
      --overwrite_output_dir \
      --predict_with_generate \
      --text_column unifiedqa-question \
      --summary_column answer \
      --seed 2023 \
      --num_train_epochs 30 \
      --train_file qulac_train.csv \
      --validation_file qulac_dev.csv \
      --test_file qulac_test.csv \

    ```

3. Run document retrieval experiments. (with post-processed t5 results for example)
    ```
    cd cosearcher
    $ python3 src/main.py --output_file_path ../src/output/t5-small-qulac-full.csv > ../src/output/t5-small-qulac-full.json
    ```
    
4. Run evaluations in `view.ipynb`
5. Significance test in `sigtest.py`
    
## Reference

Please cite the following work if you use this code repository in your work:
