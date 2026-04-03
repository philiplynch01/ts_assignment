import pandas as pd
summary_results = pd.read_csv("../summary.csv")
print(summary_results.groupby("model")[["total_time", "accuracy"]].describe().T)

results_hydra_paper = pd.read_csv("../results_hydra_paper.csv")
results_mrsqm_paper = pd.read_csv("../results_mrsqm_paper.csv")

print(results_hydra_paper[["accuracy"]].describe().T)
print(results_mrsqm_paper[["MrSQM_SFA_k5"]].describe().T)

mrsqm_accuracy = summary_results[summary_results["model"] == "mrsqm"][["dataset", "accuracy"]]
mrsqm_accuracy.rename(columns={"accuracy": "mrsqm"}, inplace=True)
hydra_accuracy = summary_results[summary_results["model"] == "hydra"][["dataset", "accuracy"]]
hydra_accuracy.rename(columns={"accuracy": "hydra"}, inplace=True)

mrsqm_reported = results_mrsqm_paper[["Dataset", "MrSQM_SFA_k5"]].copy()
mrsqm_reported.rename(columns={"MrSQM_SFA_k5": "mrsqm_reported"}, inplace=True)
mrsqm_reported.rename(columns={"Dataset": "dataset"}, inplace=True)
hydra_reported = results_hydra_paper[["dataset", "accuracy"]].copy()
hydra_reported.rename(columns={"accuracy": "hydra_reported"}, inplace=True)

comparison_df = mrsqm_accuracy.merge(hydra_accuracy, on="dataset").merge(mrsqm_reported, on="dataset").merge(hydra_reported, on="dataset")
print(comparison_df)