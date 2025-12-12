import pandas as pd
import numpy as np
import pickle
import os
import argparse

def analyze_human_data(file_path):
    print(f"Loading human data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Filter only relevant columns based on your description
    # SubjectID, AgeGroup, Trial, Choice, Accuracy_Binary, Reward_Binary, Target
    
    # We want to analyze separately for Children (Group 1) and Adults (Group 2)
    groups = {
        "Children": df[df["AgeGroup"] == 1],
        "Adults":   df[df["AgeGroup"] == 2]
    }
    
    results = {}
    
    for group_name, group_df in groups.items():
        print(f"Analyzing {group_name} (N={group_df['SubjectID'].nunique()})...")
        
        # 1. Overall Accuracy
        accuracy = group_df["Accuracy_Binary"].mean()
        
        # 2. Win-Stay / Lose-Shift (WSLS)
        # We need to compute this per subject to respect trial boundaries
        ws_probs = []
        ls_probs = []
        
        # Group by subject to shift columns safely
        # We can't just shift the whole DF because trial 1 of Subj 201 follows trial 140 of Subj 200.
        
        # Make a copy to avoid SettingWithCopy warnings
        sub_df = group_df.copy()
        sub_df["prev_choice"] = sub_df.groupby(["SubjectID", "Condition"])["Choice"].shift(1)
        sub_df["prev_reward"] = sub_df.groupby(["SubjectID", "Condition"])["Reward_Binary"].shift(1)
        
        # Filter valid trials
        valid = sub_df.dropna(subset=["prev_choice", "prev_reward"])
        
        # Win-Stay: Prev Reward = 1, Choice = Prev Choice
        wins = valid[valid["prev_reward"] == 1]
        win_stay = (wins["Choice"] == wins["prev_choice"]).mean()
        
        # Lose-Shift: Prev Reward = 0, Choice != Prev Choice
        losses = valid[valid["prev_reward"] == 0]
        lose_shift = (losses["Choice"] != losses["prev_choice"]).mean()
        
        # 3. Reversal Curve (Switch Latency)
        # We need to detect reversals: where Target(t) != Target(t-1) within a block
        sub_df["prev_target"] = sub_df.groupby(["SubjectID", "Condition"])["Target"].shift(1)
        
        # Identify reversal trials (Target changes)
        reversals = sub_df[sub_df["Target"] != sub_df["prev_target"]].dropna(subset=["prev_target"])
        reversal_indices = reversals.index
        
        traces = []
        window_before = 5
        window_after = 10
        target_len = window_before + window_after + 1  # total window length
        
        for idx in reversal_indices:
            # Check bounds in index space
            if idx - window_before < sub_df.index[0] or idx + window_after > sub_df.index[-1]:
                continue
            
            # Extract slice
            slice_df = sub_df.loc[idx - window_before : idx + window_after]
            
            # Ensure same subject and condition within the window
            subj_id = sub_df.loc[idx, "SubjectID"]
            cond = sub_df.loc[idx, "Condition"]
            if not ((slice_df["SubjectID"] == subj_id).all() and
                    (slice_df["Condition"] == cond).all()):
                continue
            
            arr = slice_df["Accuracy_Binary"].values
            # Only keep full-length windows
            if len(arr) == target_len:
                traces.append(arr)
        
        if traces:
            traces_arr = np.stack(traces, axis=0)  # all same length now
            avg_curve = np.mean(traces_arr, axis=0)
        else:
            avg_curve = np.zeros(target_len)
            
        results[group_name] = {
            "accuracy": accuracy,
            "win_stay": win_stay,
            "lose_shift": lose_shift,
            "reversal_curve": avg_curve
        }

        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Win-Stay: {win_stay:.3f}")
        print(f"  Lose-Shift: {lose_shift:.3f}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="final_reversal_learning_data.csv")
    parser.add_argument("--output", type=str, default="results/human_stats.pkl")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"Error: File {args.data_path} not found.")
    else:
        stats = analyze_human_data(args.data_path)
        
        if not os.path.exists('results'):
            os.makedirs('results')
            
        with open(args.output, 'wb') as f:
            pickle.dump(stats, f)
        print(f"Human stats saved to {args.output}")
