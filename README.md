# An In-depth Investigation of User Response Simulation for Conversational Search

## Package requirements (recommended versions).
1. python==3.8
2. `pip install -r src/requirements.txt`
3. Download data

   This repo contains a modified version of [CoSearcher](https://github.com/amzn/cosearcher). 
    ```
    cd cosearcher
    scripts/bootstrap.sh
    ```

    (Some minor errors may occur at this step.)


## How to use
1. Preprocess data and generate train/val/test split with long/short cooperativeness split.
   
    `cd src`
   
    Run `data_process.ipynb` [cell 1](https://github.com/zhenduow/UserSim/blob/main/src/data_process.ipynb)
    
3. Train the RoBERTa classifier for answer type prediction.
   
    Run `data_process.ipynb` [cell 2](https://github.com/zhenduow/UserSim/blob/main/src/data_process.ipynb) and [cell 3](https://github.com/zhenduow/UserSim/blob/main/src/data_process.ipynb)

5. Predict answer type with trained RoBERTa classifier.
    
    Run `data_process.ipynb` [cell 4](https://github.com/zhenduow/UserSim/blob/main/src/data_process.ipynb)
    
3. Finetune T5 and UnifiedQA (on Qulac for example) 
  
    ```
    cd src
    deepspeed t5trainer.py \
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
    deepspeed t5trainer.py \
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
4. Process the generated responses from LMs and save it to csv for evaluation.

   Run `data_process.ipynb` [cell 5](https://github.com/zhenduow/UserSim/blob/main/src/data_process.ipynb) with the `test_file` and `output_dir` from the last step.
3. Run document retrieval experiments using the `output_csv` from the last step. (with post-processed t5 results for example)

    ```
    cd cosearcher
    python3 src/main.py --dataset qulac --output_file_path ../src/output/t5-small-qulac.csv > ../src/output/t5-small-qulac.json
    ```
   Make sure to change `dataset` to clariq for document retrieval experiments on ClariQ.
    
5. Run evaluations in `view.ipynb`
6. Significance test in `sigtest.py`
7. Human evaluation instructions and results are available in `data/crowd`
    
## Reference

Please cite the following work if you use this code repository in your work:
```
@inproceedings{wang2024depth,
  title={An in-depth investigation of user response simulation for conversational search},
  author={Wang, Zhenduo and Xu, Zhichao and Srikumar, Vivek and Ai, Qingyao},
  booktitle={Proceedings of the ACM Web Conference 2024},
  pages={1407--1418},
  year={2024}
}
```
