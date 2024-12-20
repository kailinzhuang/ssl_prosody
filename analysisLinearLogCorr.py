# 
"""
This script is used to do linear probing of self-supervised learning (SSL) models of speech (e.g., wav2vec2.0, HuBERT, etc.) 
and prosodic prominence features (e.g., a 1-dimentional vector for each sentence).
The main analysis is to do linear probing between layers of the self-supervised learning models and prosodic prominence features.
The steps of analysis are as follows:
1. Load SSL models from Hugging Face transformers: HuBERT, wav2vec2.0, etc
2. Load prosodic prominence features: from Helsinki Prosody Corpus (HPC), both continuous and discrete prominence labels
3. Extract word level features from SSL models (with Montreal Forced Aligner alignments) https://github.com/kan-bayashi/LibriTTSLabel?tab=readme-ov-file 
4. Do linear probing analysis between SSL models and prosodic prominence features
5.
"""

# %%
import torch, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2Model, HubertModel, WavLMModel
from transformers import AutoProcessor, AutoModelForCTC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import log_loss, confusion_matrix, classification_report
import os, tqdm, librosa, pickle, glob
from praatio import textgrid
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler


# ===== Load SSL models =====
def load_ssl_model(model_name):
    if model_name == "wav2vec2":
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        layers = model.config.num_hidden_layers
        print(f"    {model_name} num_hidden_layers layers: {layers}")
        # print(f"    {model_name} encoder layers: {len(model.encoder.layers)}")
    elif model_name == "wav2vec2_large":
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        layers = model.config.num_hidden_layers
        print(f"    {model_name} encoder layers: {layers}")
    elif model_name == "hubert":
        processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        model = AutoModelForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        layers = model.config.num_hidden_layers
        print(f"    {model_name} encoder layers: {layers}")
    else:
        raise ValueError("Model not supported: choose 'wav2vec2' or 'hubert'")
    model.eval()
    return model, processor, layers

# ===== Load Prosodic Prominence Features =====
def load_prosodic_data(prosodic_prominence_fname, type="real-valued"):
    """
    Load prosodic prominence features from a file.
    Parameters:
        prosodic_prominence_fname (str): File path to the prosodic prominence features.
        type (str): Type of prosodic prominence features to load. Options: "discrete", "real-valued".
    Returns:
        prosodic_prominence_df (pd.DataFrame): DataFrame containing the prosodic prominence features.
        prominence_labels (list): List of prominence labels for each sentence.
    """
    column_names = ["word", "discrete_prominence_label", "discrete_word_boundary_label", 
                    "real-valued_prominence_label", "real-valued_word_boundary_label"]
    prosodic_prominence_df = pd.read_csv(prosodic_prominence_fname, sep="\t", names=column_names)
    sentences = prosodic_prominence_df.groupby(prosodic_prominence_df.index // 10) 
    if type == "discrete":
        prominence_labels = [sentence["discrete_prominence_label"].to_numpy() for _, sentence in sentences]
        boundary_labels = [sentence["discrete_word_boundary_label"].to_numpy() for _, sentence in sentences]
    elif type == "real-valued":
        prominence_labels = [sentence["real-valued_prominence_label"].to_numpy() for _, sentence in sentences]
        boundary_labels = [sentence["real-valued_word_boundary_label"].to_numpy() for _, sentence in sentences]
    return prosodic_prominence_df, prominence_labels, boundary_labels

def get_file_paths(base_dir, file_extension=".wav"):
    audio_paths = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(file_extension):
                file_key = os.path.splitext(file)[0]  # Filename without extension
                audio_paths[file_key] = os.path.join(root, file)
    return audio_paths

# load alignment textgrid files from Montreal Forced Aligner
def load_prosodic_data_and_audio_nested_with_alignment(prosodic_prominence_fname, base_audio_dir, alignment_dir, target_sr=16000):
    column_names = ["word", "discrete_prominence_label", "discrete_word_boundary_label", 
                    "real-valued_prominence_label", "real-valued_word_boundary_label", "filename"]
    prosodic_prominence_df = pd.read_csv(prosodic_prominence_fname, sep="\t", names=column_names)
    
    audio_paths = get_file_paths(base_audio_dir)

    filenames = prosodic_prominence_df["discrete_prominence_label"].unique()
    filenames = [x for x in filenames if str(x).endswith(".txt")]
    for i in range(len(prosodic_prominence_df)):
        label = str(prosodic_prominence_df.iloc[i]["discrete_prominence_label"])
        if label.endswith(".txt"):
            filename = label
            prosodic_prominence_df.at[i, "filename"] = filename
        else:
            if not label.endswith(".txt"):
                prosodic_prominence_df.at[i, "filename"] = prosodic_prominence_df.iloc[i-1]["filename"]

    audio_signals = []
    transcripts = []
    for filename in tqdm.tqdm(filenames):
        file_key = os.path.splitext(filename)[0]
        if file_key in audio_paths:
            signal, sr = librosa.load(audio_paths[file_key], sr=target_sr)
            transcript_path = f"{audio_paths[file_key][:-4]}.original.txt"
            transcript = open(transcript_path).read().strip()
            audio_signals.append(signal)
            transcripts.append(transcript)
        else:
            print(f"Warning: Audio file {filename} not found in {base_audio_dir}")
    
    og_prominence_labels = [
        group["real-valued_prominence_label"].dropna().to_numpy()
        for _, group in prosodic_prominence_df.groupby("filename")]
    og_boundary_labels = [
        group["real-valued_word_boundary_label"].dropna().to_numpy()
        for _, group in prosodic_prominence_df.groupby("filename")]
    clean_prominence_labels = [x for x in og_prominence_labels if not np.isnan(x).any()]
    clean_boundary_labels = [x for x in og_boundary_labels if not np.isnan(x).any()]
    
    discrete_prominence_labels = [
        group["discrete_prominence_label"].dropna().to_numpy()[1:]
        for _, group in prosodic_prominence_df.groupby("filename")]
    discrete_boundary_labels = [
        group["discrete_word_boundary_label"].dropna().to_numpy()
        for _, group in prosodic_prominence_df.groupby("filename")]
    # clean_discrete_prominence_labels = [x for x in discrete_prominence_labels if not np.isnan(x).any()]
    # clean_discrete_boundary_labels = [x for x in discrete_boundary_labels if not np.isnan(x).any()]
    
    # with open("processed_speech_data.pkl", "wb") as f:
    #     pickle.dump((filenames, audio_signals, transcripts, clean_prominence_labels, clean_boundary_labels, 
    #                  discrete_prominence_labels, discrete_boundary_labels), f)
    print(f"    number of audio signals: {len(audio_signals)}. saved to processed_speech_data.pkl")
    
    print(f"    create dataframe...")
    df = pd.DataFrame(columns=["filenames", "audio_signals", "transcripts", "prominence_vals", "words", "alignments"])
    df['filenames'] = [x[:-4] for x in filenames]
    df["audio_signals"] = audio_signals
    df["transcripts"] = transcripts
    df["prominence_vals"] = og_prominence_labels
    df["boundary_vals"] = og_boundary_labels
    df["discrete_prominence_vals"] = discrete_prominence_labels
    df["discrete_boundary_vals"] = discrete_boundary_labels

    for i, row in tqdm.tqdm(df.iterrows()):
        filename = row["filenames"]
        _sid = filename.split("_")[0]
        _chid = filename.split("_")[1]
        alignment_path = f"{alignment_dir}/{_sid}/{_chid}/{filename}.TextGrid"
        tg = textgrid.openTextgrid(alignment_path, False)
        wordTier = tg.getTier('words')
        df.at[i, "words"] = [entry.label for entry in wordTier.entries]
        df.at[i, "alignments"] = wordTier.entries
    with open("processed_speech_data_df.pkl", "wb") as f:
        pickle.dump(df, f)
    print("    done.")
    return df

# ===== Extract SSL Model Representations =====
def extract_ssl_embeddings(df, output_path, model_name, slice_size, target_sr=16000):
    print(f"\nLoading {model_name}...")
    ssl_model, ssl_processor, layers = load_ssl_model(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    ssl_model.to(device)
    ssl_model.eval()

    num_layers = ssl_model.config.num_hidden_layers
    print(f"Number of Hidden Layers: {num_layers}")

    total_rows = len(df)
    num_slices = (total_rows + slice_size - 1) // slice_size
    print(f"number of slices {num_slices}")
    for slice_idx in range(num_slices):
        first_slice = slice_idx * slice_size
        last_slice = min(first_slice + slice_size, total_rows)
        print(f"Processing slice {first_slice} to {last_slice}...")
        fname = f"{output_path}/processed_speech_data_df_ssl_representations_{model_name}_{first_slice}_{last_slice}.pkl"
        if os.path.exists(fname):
            print(f"    File {fname} already exists. Skipping...")
            continue

        slice_df = df[first_slice:last_slice]

        slice_df["word_level_rep"] = None
        slice_df["concat_rep"] = None
        all_concat_rep = []
        all_prominence = []

        for i, row in tqdm.tqdm(slice_df.iterrows(), total=len(slice_df)):
            audio_signal = torch.tensor(row["audio_signals"]).unsqueeze(0).to(device)
            alignments = row["alignments"]

            with torch.no_grad():
                outputs = ssl_model(audio_signal, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # (layer, batch, time, feature_dim)

            # Convert to (layer, time, feature_dim)
            hidden_states = torch.stack(hidden_states, dim=0).squeeze(1)  # (num_layers + 1, time, feature_dim)

            word_representations = []
            for alignment in alignments:
                start, end, word = alignment
                if word == "":
                    continue

                start_idx = int(start * target_sr / ssl_model.config.hidden_size)
                end_idx = int(end * target_sr / ssl_model.config.hidden_size)

                start_idx = max(0, start_idx)
                end_idx = min(hidden_states.size(1), end_idx)

                word_frames = hidden_states[:, start_idx:end_idx, :]  # (num_layers + 1, num_frames, feature_dim)
                if word_frames.size(1) > 0:
                    word_avg = word_frames.mean(dim=1)  # average across frames -> (num_layers + 1, feature_dim)
                else:
                    word_avg = torch.zeros(hidden_states.size(0), hidden_states.size(2))

                word_representations.append({"word": word, "representation": word_avg.cpu().numpy()})

            representations = [entry["representation"] for entry in word_representations]
            concat_rep = np.stack(representations, axis=1)
            all_concat_rep.append(concat_rep)
            all_prominence.append(row["prominence_vals"])

            slice_df.at[i, "word_level_rep"] = word_representations
            slice_df.at[i, "concat_rep"] = concat_rep

            save_df = slice_df[["filenames", "word_level_rep", "concat_rep"]]

        with open(fname, "wb") as f:
            pickle.dump(save_df, f)
        print(f"    Slice {first_slice} to {last_slice} saved.")

        del slice_df, all_concat_rep, all_prominence
        gc.collect()
        torch.cuda.empty_cache()
    print("     SSL extract embeddings complete.")

def combine_processed_slices_small(input_path, model_name, chunk_size=1):
    slice_files = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.startswith(f"processed_speech_data_df_ssl_representations_{model_name}") and f.endswith(".pkl")
    ]
    print(f"Found {len(slice_files)} slice files to combine.")
    rep_df = pd.DataFrame(columns=["filenames", "concat_rep"])
    for i, file in enumerate(slice_files):
      print(f"  Loading file {i}...")
      with open(file, "rb") as f:
          _load_df = pickle.load(f)
      rep_df = pd.concat([rep_df, _load_df[["filenames", "concat_rep"]]], ignore_index=True)
      del _load_df

    with open(f"{output_path}/processed_speech_data_df_ssl_representations_{model_name}_combined_df.pkl", "wb") as f:
        pickle.dump(rep_df, f)


# ===== Linear Probing =====

def get_prosody_labels(df, ssl_df):
    df_analysis = df[["filenames", "prominence_vals", "boundary_vals", 
                  "discrete_prominence_vals", "discrete_boundary_vals"]]
    df_analysis["concat_rep"] = None
    not_found = []
    for i, row in df_analysis.iterrows():
        filename = row["filenames"]
        ssl_row = ssl_df[ssl_df["filenames"] == filename]
        if len(ssl_row) > 0:
            df_analysis.at[i, "concat_rep"] = ssl_row["concat_rep"].values[0]
        else:
            not_found.append(filename)

    df_analysis = df_analysis.dropna(subset=["concat_rep"])

    df_analysis["match_length"] = None
    non_match_count = 0
    for i, row in df_analysis.iterrows():
        if row["concat_rep"].shape[1] != len(row["boundary_vals"]) or row["concat_rep"].shape[1] != len(row["prominence_vals"]) or row["concat_rep"].shape[1] != len(row["discrete_prominence_vals"]) or row["concat_rep"].shape[1] != len(row["discrete_boundary_vals"]):
            df_analysis.at[i, "match_length"] = False
            non_match_count += 1
        else:
            df_analysis.at[i, "match_length"] = True
    df_analysis = df_analysis[df_analysis["match_length"] == True]

    all_rep = df_analysis["concat_rep"].to_numpy()

    all_prominence = df_analysis["prominence_vals"].to_numpy()
    all_boundary = df_analysis["boundary_vals"].to_numpy()
    all_prominence = np.hstack(all_prominence)
    all_boundary = np.hstack(all_boundary)

    discrete_prominence = df_analysis["discrete_prominence_vals"].to_numpy()
    discrete_boundary = df_analysis["discrete_boundary_vals"].to_numpy()
    discrete_boundary = np.hstack(discrete_boundary)
    discrete_prominence = np.hstack(discrete_prominence)
    return all_rep,all_prominence,all_boundary,discrete_prominence,discrete_boundary

def linear_probe_word_level(all_rep, labels, layers, do_pca=False):
    r2_scores = np.zeros((25,))
    mse_scores = np.zeros((25,))
    for layer in range(layers+1):    
        stacked_data = []
        for rep in all_rep:
            stacked_data.append(rep[layer])
        final_array = np.vstack(stacked_data)
        
        X = final_array
        y = labels
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)
        reg = LinearRegression().fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2_scores[layer] = r2
        mse_scores[layer] = mse
        print(f"    layer {layer} r2: {round(r2, 2)}, mse: {round(mse, 2)}")
    return r2_scores, mse_scores
def compute_pearsr(X, y):
    X_mean = np.mean(X, axis=0)  # Mean of each feature across words (shape: 1024,)
    y_mean = np.mean(y)          # Mean of the target (scalar)

    # Center data
    X_centered = X - X_mean      # Center features (shape: 96965, 1024)
    y_centered = y - y_mean      # Center target (shape: 96965,)

    
    # Compute standard deviations
    X_std = np.std(X, axis=0)  # Standard deviation of each feature (shape: 1024,)
    y_std = np.std(y)          # Standard deviation of the target (scalar)

    X_normalized = X_centered / (X_std + 1e-8)
    y_normalized = y_centered / (y_std + 1e-8)
    # Compute the dot product between X and y for covariance
    covariance = np.dot(X_normalized.T, y_normalized) / len(y)

    # Pearson correlation for each feature
    pearson_corr = covariance
    return pearson_corr
    
def correlation_word_level(model_name, output_path, layers, all_rep, all_prominence, all_boundary, compute_pearsr):
    all_prominence_shuffled = shuffle(all_prominence)
    all_boundary_shuffled = shuffle(all_boundary)
    corrs_prominence = []
    corrs_boundary = []
    corrs_prominence_shuffled = []
    corrs_boundary_shuffled = []
    for layer in range(layers+1):
        stacked_data = []
        for rep in all_rep:
            stacked_data.append(rep[layer])
        final_array = np.vstack(stacked_data)
        X = final_array
        y_prominence = all_prominence
        y_boundary = all_boundary
        y_prominence_shuffled = all_prominence_shuffled
        y_boundary_shuffled = all_boundary_shuffled
    # print(f"X shape: {X.shape}, y shape: {y_prominence.shape}")
    
        corrs_prominence.append(compute_pearsr(X, y_prominence))
        corrs_boundary.append(compute_pearsr(X, y_boundary))
        corrs_prominence_shuffled.append(compute_pearsr(X, y_prominence_shuffled))
        corrs_boundary_shuffled.append(compute_pearsr(X, y_boundary_shuffled))
        print(f"    layer {layer} done.")
        
    corr_df = pd.DataFrame({"r_prominence": (corrs_prominence), "r_boundary": corrs_boundary, "r_prominence_shuffled": corrs_prominence_shuffled, "r_boundary_shuffled": corrs_boundary_shuffled})

    for i, row in corr_df.iterrows():
        corr_df.at[i, "r_prominence_mean"] = row["r_prominence"].mean()
        corr_df.at[i, "r_boundary_mean"] = row["r_boundary"].mean()
        corr_df.at[i, "r_prominence_shuffled_mean"] = row["r_prominence_shuffled"].mean()
        corr_df.at[i, "r_boundary_shuffled_mean"] = row["r_boundary_shuffled"].mean()
    corr_df.to_csv(f"{output_path}/pearson_corr_{model_name}.csv", index=False)
    return corrs_prominence,corrs_boundary,corr_df

def logistic_reg_word_level(layers, all_rep, discrete_labels, binary=False):
    metrics = pd.DataFrame(columns=["layer", "accuracy", "precision", "recall", "f1", "roc_auc", "logloss", "conf_matrix", "mcfadden_r2", "report"])
    for layer in range(layers+1):
        stacked_data = []
        for rep in all_rep:
            stacked_data.append(rep[layer])
        final_array = np.vstack(stacked_data)
        X = final_array
        y = discrete_labels
        if binary:
            y = np.where(y > 0, 1, 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2024)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train, X_test = X_train_scaled, X_test_scaled
        
        clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        # clf = LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs')
        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]  # Predicted probabilities for class 1

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        logloss = log_loss(y_test, y_prob)
        conf_matrix = confusion_matrix(y_test, y_pred)
        # Pseudo R^2 (McFadden's)
        def pseudo_r2(clf, X, y):
            ll_model = -log_loss(y, clf.predict_proba(X), normalize=False)
            y_mean = np.mean(y)
            ll_null = -log_loss(y, np.full_like(y, y_mean), normalize=False)
            return 1 - (ll_model / ll_null)
        mcfadden_r2 = pseudo_r2(clf, X_test, y_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        row = {
            "layer": layer,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "logloss": logloss,
            "conf_matrix": conf_matrix,
            "mcfadden_r2": mcfadden_r2,
            "report": report
        }
        metrics = pd.concat([metrics, pd.DataFrame([row])], ignore_index=True)
        print(f"   layer {layer} done.")
    return metrics

# ===== Plot =====
def plot_linear_r2(model_name, linear_result_df):
    r2_prominence = linear_result_df["r2_prominence"].to_numpy()
    r2_boundary = linear_result_df["r2_boundary"].to_numpy()
    r2_prominence_shuffled = linear_result_df["r2_prominence_shuffled"].to_numpy()
    r2_boundary_shuffled = linear_result_df["r2_boundary_shuffled"].to_numpy()
    mse_prominence = linear_result_df["mse_prominence"].to_numpy()
    mse_boundary = linear_result_df["mse_boundary"].to_numpy()
    mse_prominence_shuffled = linear_result_df["mse_prominence_shuffled"].to_numpy()
    mse_boundary_shuffled = linear_result_df["mse_boundary_shuffled"].to_numpy()
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 2), dpi=300)
    ax.plot(r2_prominence, "-o", label="prominence", color="red", linewidth=2)
    ax.plot(r2_boundary, "-o", label="boundary", color="blue", linewidth=2)
    ax.plot(r2_prominence_shuffled, "-.", label="prominence (shuffled)", color="red", linewidth=1.5)
    ax.plot(r2_boundary_shuffled, "-.", label="boundary (shuffled)", color="blue", linewidth=1.5)
    ax.set_xticks(range(len(r2_prominence)))
    ax.set_title(f"{model_name}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("R2 Score")
    ax.set_yticks(np.arange(-0.01, 0.06, 0.01))
    fig.tight_layout()
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 2), dpi=300)
    ax.plot(mse_prominence, "-o", label="prominence", color="red", linewidth=2)
    ax.plot(mse_boundary, "-o", label="boundary", color="blue", linewidth=2)
    ax.plot(mse_prominence_shuffled, "-.", label="prominence (shuffled)", color="red", linewidth=1.5)
    ax.plot(mse_boundary_shuffled, "-.", label="boundary (shuffled)", color="blue", linewidth=1.5)
    ax.set_xticks(range(len(mse_prominence)))
    ax.set_title(f"{model_name}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("MSE")
    fig.tight_layout()

def plot_metric(ax, layers, metrics_df1, metrics_df2, label_df1, label_df2, col_df1, col_df2, column, title, 
                legend=False, grid=False):
    ax.plot(layers, np.array(metrics_df1[column]), marker='o', color=col_df1, label=label_df1)
    ax.plot(layers, np.array(metrics_df2[column]), marker='o', color=col_df2, label=label_df2)
    ax.set_title(f"{title} per Layer")
    ax.set_xlabel('Layer')
    ax.set_ylabel(f"{title}")
    if legend:
        ax.legend()
    if grid:
        ax.grid(True)
    ax.set_xticks(layers)
    ax.set_xticklabels(layers)
    ax.set_ylim(min(min(metrics_df1[column]), min(metrics_df2[column])) - 0.05,
                max(max(metrics_df1[column]), max(metrics_df2[column])) + 0.05)

def plot_corrs(model_name, corrs_prominence, corrs_boundary, corr_df):
    fig, ax = plt.subplots(1, 1, figsize=(6, 2), dpi=300)
    # ax.plot(corrs_prominence, label="prominence", color="red", alpha=0.008)
    # ax.plot(corrs_boundary, label="boundary", color="blue", alpha=0.008)
    ax.plot(corr_df["r_prominence_mean"], label="prominence (mean)", color="red", linestyle="-")
    ax.plot(corr_df["r_boundary_mean"], label="boundary (mean)", color="blue", linestyle="-")
    # ax.plot(corrs_prominence_shuffled, label="prominence (shuffled)", color="red", linestyle="--", alpha=0.1)
    ax.set_xticks(range(len(corrs_prominence)))
    ax.set_title(f"{model_name}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pearson Correlation")
    ax.set_ylim(min(min(corr_df["r_prominence_mean"]), min(corr_df["r_boundary_mean"])) - 0.05,
                max(max(corr_df["r_prominence_mean"]), max(corr_df["r_boundary_mean"])) + 0.05)
    ax.legend()
    fig.tight_layout()
    
def plot_metrics_comparison(fig, axs, prominence_data, boundary_data, layers, ):
    metrics_df1 = prominence_data
    metrics_df2 = boundary_data
    label_df1 = "Prominence"
    label_df2 = "Boundary"
    col_df1 = "r"
    col_df2 = "b"
    
    layers = np.arange(layers+1)
    plot_metric(axs[0, 0], layers, metrics_df1, metrics_df2, label_df1, label_df2, col_df1, col_df2, "accuracy", "Accuracy", legend=True)
    plot_metric(axs[0, 1], layers, metrics_df1, metrics_df2, label_df1, label_df2, col_df1, col_df2, "roc_auc", "ROC AUC")
    plot_metric(axs[1, 0], layers, metrics_df1, metrics_df2, label_df1, label_df2, col_df1, col_df2, "precision", "Precision")
    plot_metric(axs[1, 1], layers, metrics_df1, metrics_df2, label_df1, label_df2, col_df1, col_df2, "recall", "Recall")
    plot_metric(axs[2, 0], layers, metrics_df1, metrics_df2, label_df1, label_df2, col_df1, col_df2, "f1", "F1 Score")
    plot_metric(axs[2, 1], layers, metrics_df1, metrics_df2, label_df1, label_df2, col_df1, col_df2, "logloss", "Log Loss")
    
    fig.tight_layout()
    fig.show()
    return fig, axs
# if __name__ == "__main__":
    # Configurations
model_name = "wav2vec2"  # "wav2vec2", "hubert", "wav2vec2-large-960h", "hubert-large-ls960", "wavlm_large"
data_path = "."
prosodic_prominence_fname = f"{data_path}/dev.txt"
audio_dir = f"{data_path}/dev-clean"
alignment_dir = f"{data_path}/align-dev-clean"
aggregation = "layerwise"
# output_path = "./ssl_output"
output_path = "/Users/kailinzhuang/Downloads/final results"
ssl_representation_fname = f"representation_prominence_pairs_dev_{model_name}_{aggregation}.pkl"

need_to_extract_embeddings = False
need_to_process_data = False
need_to_slice_data = False
need_to_do_linear_regression = False
need_to_do_correlation_analysis = False
need_to_do_logistic_regression = False

# load prosodic prominence labels, audio files, and alignments
print("\nLoading prosodic prominence and audio files...")
if need_to_process_data:
    print("    process data...")
    df = load_prosodic_data_and_audio_nested_with_alignment(prosodic_prominence_fname, audio_dir, alignment_dir)
else: 
    with open(f"{output_path}/processed_speech_data_df.pkl", "rb") as f:
        df = pickle.load(f)

# extract word level SSL representation
print(f"\nGet SSL model  word level representations for {model_name.capitalize()}...")
if need_to_extract_embeddings:
    extract_ssl_embeddings(df, output_path, model_name, slice_size=1000, target_sr=16000)
    print("    combine df slices, save only concat_rep...")
    combine_processed_slices_small(output_path, model_name)
else:
    if model_name == "wav2vec2":
        layers = 12
    elif model_name == "hubert":
        layers = 24
    elif model_name == "wav2vec2_large":
        layers = 24
    fname = f"{output_path}/processed_speech_data_df_ssl_representations_{model_name}_combined_df.pkl"
    print("     already processed, load ssl df")
    with open(fname, "rb") as f:
        ssl_df = pickle.load(f)

print("\nGet prosody labels...")
all_rep, all_prominence, all_boundary, discrete_prominence, discrete_boundary = get_prosody_labels(df, ssl_df)

# perform linear regression
if need_to_do_linear_regression:
    print("   do linear regression")
    print("     prominence")
    r2_prominence, mse_prominence = linear_probe_word_level(all_rep, all_prominence, layers=layers)
    print("     boundary")
    r2_boundary, mse_boundary = linear_probe_word_level(all_rep, all_boundary, layers=layers)

    print("   shuffle prominence and boundary labels")
    all_prominence_shuffled = shuffle(all_prominence)
    all_boundary_shuffled = shuffle(all_boundary)
    print("     prominence")
    r2_prominence_shuffled, mse_prominence_shuffled = linear_probe_word_level(all_rep, all_prominence_shuffled, layers=layers)
    print("     boundary")
    r2_boundary_shuffled, mse_boundary_shuffled = linear_probe_word_level(all_rep, all_boundary_shuffled, layers=layers)

    save_df = pd.DataFrame({"r2_prominence": r2_prominence, "mse_prominence": mse_prominence, 
                            "r2_boundary": r2_boundary, "mse_boundary": mse_boundary,
                            "r2_prominence_shuffled": r2_prominence_shuffled, "mse_prominence_shuffled": mse_prominence_shuffled, 
                            "r2_boundary_shuffled": r2_boundary_shuffled, "mse_boundary_shuffled": mse_boundary_shuffled})
    save_df.to_csv(f"{output_path}/linear_probing_results_{model_name}.csv", index=False)
else:
    print(f"\nLoad linear regression results for {model_name}")
    linear_result_df = pd.read_csv(f"{output_path}/linear_probing_results_{model_name}.csv")

plot_linear_r2(model_name, linear_result_df)
# compare r2 scores, p value
r2_boundary = linear_result_df["r2_boundary"].to_numpy()
r2_prominence = linear_result_df["r2_prominence"].to_numpy()
pval = np.zeros((25,))
from scipy.stats import ttest_ind
for i in range(layers):
    t, p = ttest_ind(r2_prominence[i], r2_boundary[i])
    pval[i] = p
    

print("\nCorrelation analysis")
if need_to_do_correlation_analysis:
    corrs_prominence, corrs_boundary, corr_df = correlation_word_level(model_name, output_path, layers, all_rep, all_prominence, all_boundary, compute_pearsr)
else:
    corr_df = pd.read_csv(f"{output_path}/pearson_corr_{model_name}.csv")
    corrs_prominence = corr_df["r_prominence_mean"].to_numpy()
    corrs_boundary = corr_df["r_boundary_mean"].to_numpy()
plot_corrs(model_name, corrs_prominence, corrs_boundary, corr_df)

if need_to_do_logistic_regression:
    binary = True
    discrete_prominence = pd.to_numeric(discrete_prominence, errors='coerce')
    print("     prominence")
    logistic_metrics_prominence = logistic_reg_word_level(layers, all_rep, discrete_prominence, binary=binary)
    print("     boundary")
    logistic_metrics_boundary = logistic_reg_word_level(layers, all_rep, discrete_boundary, binary=binary)

    print("     saving logistic regression results")
    print("     prominence")
    logistic_metrics_prominence.to_csv(f"{output_path}/logistic_regression_prominence_{model_name}_binary_{binary}.csv", index=False)
    print("     boundary")
    logistic_metrics_boundary.to_csv(f"{output_path}/logistic_regression_boundary_{model_name}_binary_{binary}.csv", index=False)
    print("     done.")
else:
    print("     loading logistic regression results")
    logistic_metrics_prominence = pd.read_csv(f"{output_path}/logistic_regression_prominence_{model_name}_binary_True.csv")
    logistic_metrics_boundary = pd.read_csv(f"{output_path}/logistic_regression_boundary_{model_name}_binary_True.csv")
    print("     done.")


fig, axs = plt.subplots(3, 2, figsize=(12, 6), dpi=300)
plot_metrics_comparison(fig, axs, prominence_data=logistic_metrics_prominence, boundary_data=logistic_metrics_boundary, layers=layers)
fig.suptitle(f"Logistic Regression Metrics per Layer ({model_name})", y=1.05)
