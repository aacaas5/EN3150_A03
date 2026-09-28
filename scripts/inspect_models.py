import torch
import pandas as pd
from config import ROOT
from src.models import build_model
from src.profiling import architecture_table, convolution_savings
from src.utils import seed_everything

def inspect():
    seed_everything()
    rows = []
    for name in ("model_a", "model_b", "mobilenet_v2", "efficientnet_b0"):
        model = build_model(name)
        model.eval()
        with torch.no_grad():
            assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 10)
        rows.append({"model": name, **architecture_table(model, name)})
    pd.DataFrame(rows).to_csv(ROOT / "results/tables/model_inspection.csv", index=False)
    convolution_savings()
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

if __name__ == "__main__":
    inspect()
