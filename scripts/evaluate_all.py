import pandas as pd
from config import ROOT
from src.evaluation import evaluate
from src.plotting import final_plots

def run():
    df = pd.DataFrame([evaluate(name) for name in ("model_a", "model_b", "mobilenet_v2", "efficientnet_b0")])
    df.to_csv(ROOT / "results/tables/final_comparison.csv", index=False)
    df.iloc[:2].to_csv(ROOT / "results/tables/custom_models.csv", index=False)
    df.iloc[2:].to_csv(ROOT / "results/tables/pretrained_models.csv", index=False)
    final_plots(df)
    print(df.to_string(index=False), flush=True)

if __name__ == "__main__":
    run()
