# Running the experiments on the college PC

Everything here is **exploration on non-T4 hardware**. It must be labelled "non-T4" in the report, and the submitted adapter and the report's
tables still come from the Colab T4.

## 1. Unpack
Unzip **both** zips into the SAME folder, so you get:
```
some_folder/
  nimbus_code/       (run_college.py, src/, audit/)
  candidate_pack/    (data/train.jsonl, data/dev.jsonl, score.py, SCHEMA.md)   <- company data: private
```

## 2. Check the GPU and libraries (30 seconds)
```
python -c "import torch, transformers, peft; print(torch.cuda.get_device_name(0), torch.__version__, transformers.__version__, peft.__version__)"
```
If `peft` later fails with an error about `torchao`, run `pip uninstall -y torchao`.

## 3. Run
```
cd nimbus_code
python run_college.py --pack ../candidate_pack --jobs R0_repeat,R0b_seed1,D3b_5epochs,D4_keyorder
```
Other jobs: `D3a_1epoch, D5_rank8, D5_rank32, D6_lr1e-4, D6_lr4e-4, D2_qlora4` (QLoRA needs bitsandbytes).
The model (Qwen/Qwen2.5-1.5B-Instruct, about 3 GB) downloads from Hugging Face on first use. With no internet, pass `--model <local folder>`.
One failing job does not stop the others. Order the list by importance, because the GPU may be shared.

## 4. Send the results back
When it prints `DONE`, copy this file to the laptop:
```
nimbus_code/college_out/college_results.zip
```
It contains the adapters (PEFT format), dev predictions, `results.jsonl`, `summary.csv`, loss logs and `environment.json`. It contains **no company data**.

## 5. Clean up (condition set by the company)
Delete the `candidate_pack/` folder and `college_out/` (after copying) from the college PC, empty the recycle bin, and sign out of any accounts
you used. The Hugging Face model cache does not contain company data.
