import os
import re
import glob
import argparse


import wandb
import torch
import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from eval.inference_func import inference
from tqdm import tqdm


def do_inference(config_path_data, config_path_model, config_path_trainer, wandb_run_id: str, batch_size: int) -> None:
    config_data = OmegaConf.load(config_path_data)
    config_model = OmegaConf.load(config_path_model)
    config_trainer = OmegaConf.load(config_path_trainer)

    config = OmegaConf.merge(config_data, config_model, config_trainer)

    orig_exp_title = config.title
    # in title replace "-(bs-16)-" with "-(bs-{batch_size})-"
    exp_title = re.sub(r"-\(bs-\d+\)-", f"-(bs-{batch_size})-", orig_exp_title)

    title_dir = exp_title.replace("/", "-").replace(" ", "_")
    wandb_runs_dir = config.logger.get("runs_dir", "./experiments")

    checkpoint_paths = glob.glob(os.path.join("./", config.logger.wandb.project, str(wandb_run_id), "**", "*.ckpt"), recursive=True)
    # remove "last.ckpt" from the list
    checkpoint_paths = [path for path in checkpoint_paths if "last.ckpt" not in path]
    assert len(checkpoint_paths) == 1
    checkpoint_path = checkpoint_paths[0]

    results, predict_mean_scores, true_mean_scores, sys_pred_means, sys_true_means = inference(
        config=config,
        checkpoint_path=checkpoint_path,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    # write results to a csv file and formated in a txt file
    results_df = pd.DataFrame(results, columns=["sys_mse", "sys_lcc", "sys_srcc", "sys_ktau", "utt_mse", "utt_lcc", "utt_srcc", "utt_ktau"])
    results_df["sys_mse"] = [results["system_level"]["MSE"]]
    results_df["sys_lcc"] = [results["system_level"]["LCC"]]
    results_df["sys_srcc"] = [results["system_level"]["SRCC"]]
    results_df["sys_ktau"] = [results["system_level"]["KTAU"]]
    results_df["utt_mse"] = [results["utterance_level"]["MSE"]]
    results_df["utt_lcc"] = [results["utterance_level"]["LCC"]]
    results_df["utt_srcc"] = [results["utterance_level"]["SRCC"]]
    results_df["utt_ktau"] = [results["utterance_level"]["KTAU"]]

    sys_level_str = f"System-level Results:\t{results['system_level']['MSE']:.3f}\t{results['system_level']['LCC']:.3f}\t{results['system_level']['SRCC']:.3f}\t{results['system_level']['KTAU']:.3f}\n"
    utt_level_str = f"Utterance-level Results:\t{results['utterance_level']['MSE']:.3f}\t{results['utterance_level']['LCC']:.3f}\t{results['utterance_level']['SRCC']:.3f}\t{results['utterance_level']['KTAU']:.3f}\n"

    predictions_utt = pd.DataFrame({
        "predicted_mean_scores": predict_mean_scores,
        "true_mean_scores": true_mean_scores
    })

    predictions_sys = pd.DataFrame({
        "predicted_sys_mean_scores": sys_pred_means,
        "true_sys_mean_scores": sys_true_means
    })

    print(f"Results for experiment: {exp_title} (WandB Run ID: {wandb_run_id})")
    print(sys_level_str)
    print(utt_level_str)

    output_dir = os.path.join(wandb_runs_dir, title_dir)
    results_path = os.path.join(output_dir, "results.csv")

    predict_utt_path = os.path.join(output_dir, "predictions_utt.csv")
    predict_sys_path = os.path.join(output_dir, "predictions_sys.csv")
    txt_results_path = os.path.join(output_dir, "results.txt")

    assert os.path.exists(output_dir), f"Output directory does not exist: {output_dir}"
    assert os.path.exists(results_path), f"Results path does not exist: {results_path}"
    assert os.path.exists(predict_utt_path), f"Predictions utt path does not exist: {predict_utt_path}"
    assert os.path.exists(predict_sys_path), f"Predictions sys path does not exist: {predict_sys_path}"
    assert os.path.exists(txt_results_path), f"TXT results path does not exist: {txt_results_path}"

    # save files
    results_df.to_csv(results_path, index=False)
    predictions_utt.to_csv(predict_utt_path, index=False)
    predictions_sys.to_csv(predict_sys_path, index=False)
    with open(txt_results_path, "w") as f:
        f.write("Metric\tMSE\tLCC\tSRCC\tKTAU\n")
        f.write(sys_level_str)
        f.write(utt_level_str)


def verify_duplicates(experiments_folders: list, models: list, datasets: list, sufix: str) -> None:
    # Convert model ids to the slug form used in folder names (example shows "/" -> "-")
    model_slugs = [m.replace("/", "-") for m in models]

    model_alt = "|".join(map(re.escape, model_slugs))
    dataset_alt = "|".join(map(re.escape, datasets))

    float_re = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

    pattern = rf"""
    ^{re.escape(sufix)}
    (?P<model_name>{model_alt})
    -
    (?P<dataset_name>{dataset_alt})
    -\(FreezeBackbone-(?P<freeze_backbone>True|False)\)
    -\(LayerStrategy-(?P<layer_strategy>[^)]*)\)
    -\(epochs-(?P<epochs>\d+)\)
    -\(bs-(?P<batch_size>\d+)\)
    -\(LR-(?P<learning_rate>{float_re})\)
    $
    """

    rx = re.compile(pattern, re.VERBOSE)

    exp_dict = {}

    for exp in tqdm(experiments_folders):
        m = rx.fullmatch(exp)
        if not m:
            continue  # skip non-matching folders

        model_name = m.group("model_name")
        dataset_name = m.group("dataset_name")
        freeze_backbone = (m.group("freeze_backbone") == "True")
        layer_strategy = m.group("layer_strategy")
        epochs = int(m.group("epochs"))
        batch_size = int(m.group("batch_size"))
        learning_rate = float(m.group("learning_rate"))

        key_name = f"{model_name}-{dataset_name}-{freeze_backbone}-{layer_strategy}"

        if key_name not in exp_dict:
            exp_dict[key_name] = 0
        exp_dict[key_name] += 1

    # check for duplicates
    for key_name, count in exp_dict.items():
        if count > 1:
            print(f"Warning: Duplicate experiments found for {key_name} ({count} times)")
    else:
        print("No duplicate experiments found.")


def main() -> None:
    config_base_dir = "/hadatasets/alef.ferreira/MOS-Prediction/CAL-MOS/config"
    trainer_config_path = "/hadatasets/alef.ferreira/MOS-Prediction/CAL-MOS/config/trainer/default-16bs.yaml"
    experiments_base_dir = "/hadatasets/alef.ferreira/MOS-Prediction/CAL-MOS/src/experiments_ft"
    sufix = "CAL-MOS-"

    # models = [
    #     "facebook/wav2vec2-large",
    #     "facebook/wav2vec2-xls-r-300m",
    #     "facebook/wav2vec2-xls-r-1b",
    #     "facebook/mms-300m",
    #     "facebook/mms-1b",
    #     "facebook/w2v-bert-2.0",
    #     "facebook/hubert-large-ll60k",
    #     "microsoft/wavlm-large",
    #     "facebook/data2vec-audio-large",
    #     "openai/whisper-large",
    # ]

    models = [
        "microsoft/wavlm-large",
        "facebook/mms-300m",
        "facebook/wav2vec2-xls-r-300m",
        "facebook/w2v-bert-2.0",
        "facebook/hubert-large-ll60k",
    ]

    datasets = ["SingMOS", "BRSpeech", "TMHINT-QI", "BVCC"]

    model_dict = {
        "facebook-wav2vec2-large": "wav2vec2-large.yaml",
        "facebook-wav2vec2-xls-r-300m": "xlsr-300m.yaml",
        "facebook-wav2vec2-xls-r-1b": "xlsr-1b.yaml",
        "facebook-mms-300m": "mms-300m.yaml",
        "facebook-mms-1b": "mms-1b.yaml",
        "facebook-w2v-bert-2.0": "wav2bert2.yaml",
        "facebook-hubert-large-ll60k": "hubert-large.yaml",
        "microsoft-wavlm-large": "wavlm-large.yaml",
        "facebook-data2vec-audio-large": "data2vec-large.yaml",
        "openai-whisper-large": "whisperv3-large.yaml",
    }

    dataset_dict = {
        "SingMOS": "singmos.yaml",
        "BRSpeech": "brspeech.yaml",
        "TMHINT-QI": "tmhint-qi.yaml",
        "BVCC": "bvcc.yaml",
    }

    strategy_dict = {
        "FT": "fine-tuning",
        "LP": "frozen-last-layer",
        "weighted-sum": "weighted-sum",
    }


    model_slugs = [m.replace("/", "-") for m in models]

    model_alt = "|".join(map(re.escape, model_slugs))
    dataset_alt = "|".join(map(re.escape, datasets))

    float_re = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

    pattern = rf"""
    ^{re.escape(sufix)}
    (?P<model_name>{model_alt})
    -
    (?P<dataset_name>{dataset_alt})
    -\(FreezeBackbone-(?P<freeze_backbone>True|False)\)
    -\(LayerStrategy-(?P<layer_strategy>[^)]*)\)
    -\(epochs-(?P<epochs>\d+)\)
    -\(bs-(?P<batch_size>\d+)\)
    -\(LR-(?P<learning_rate>{float_re})\)
    $
    """

    rx = re.compile(pattern, re.VERBOSE)

    experiments_folders = os.listdir(experiments_base_dir)

    verify_duplicates(experiments_folders, models, datasets, sufix)

    # print(experiments_folders)

    flag = 0
    for exp_folder in tqdm(experiments_folders):
        m = rx.fullmatch(exp_folder)
        if not m:
            print(f"Skipping non-matching folder: {exp_folder}")
            continue  # skip non-matching folders

        model_name = m.group("model_name")
        dataset_name = m.group("dataset_name")
        freeze_backbone = (m.group("freeze_backbone") == "True")
        layer_strategy = m.group("layer_strategy")
        epochs = int(m.group("epochs"))
        batch_size = int(m.group("batch_size"))
        learning_rate = float(m.group("learning_rate"))

        # print(f"Running inference for experiment: {exp_folder}")
        # print(f"Model: {model_name}, Dataset: {dataset_name}, Freeze Backbone: {freeze_backbone}, Layer Strategy: {layer_strategy}, Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")

        if freeze_backbone==False:
            strategy_key = "FT"
        elif freeze_backbone==True and layer_strategy=="per_layer":
            strategy_key = "LP"
        elif freeze_backbone==True and layer_strategy=="weighted_sum":
            strategy_key = "weighted-sum"
        else:
            raise ValueError("Invalid combination of freeze_backbone and layer_strategy")

        strategy_dir = strategy_dict[strategy_key]

        config_model_path = os.path.join(config_base_dir, "model", strategy_dir, model_dict[model_name])
        config_data_path = os.path.join(config_base_dir, "data", dataset_dict[dataset_name])

        # check if config files exist
        assert os.path.exists(config_model_path), f"Model config file does not exist: {config_model_path}"
        assert os.path.exists(config_data_path), f"Data config file does not exist: {config_data_path}"

        experiment_wandb_file_path = os.path.join(experiments_base_dir, exp_folder, "wandb_run_id.txt")

        with open(experiment_wandb_file_path, "r") as f:
            # read the wandb run ids, can be multiple ids separated by new lines
            wandb_run_ids = f.read().splitlines()

        assert len(wandb_run_ids) > 0, f"No wandb run id found for experiment {exp_folder}"

        wandb_run_id = wandb_run_ids[-1]  # get the last run id

        # if model_name=="openai-whisper-large" and dataset_name=="BVCC" and layer_strategy=="weighted_sum":
        #     print(f"Skipping experiment {exp_folder} due to known issues with Whisper and BVCC with weighted-sum strategy.")
        #     flag=1
        #     print("\nWe start from here...\n")
        #     continue

        # if flag==0:
        #     print("Skipping...")
        #     continue

        # do_inference(
        #     config_path_data=config_data_path,
        #     config_path_model=config_model_path,
        #     config_path_trainer=trainer_config_path,
        #     wandb_run_id=wandb_run_id,
        #     batch_size=batch_size,
        # )




if __name__ == "__main__":
    main()
