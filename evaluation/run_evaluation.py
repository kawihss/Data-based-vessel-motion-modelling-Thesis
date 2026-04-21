import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.kinematic import ConstantVelocityModel
from evaluation.evaluator import run_evaluation, plot_horizon_error
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #Evaluate ConstantVelocityModel on all test files
    model = ConstantVelocityModel(velocity_fraction=0.4)
    metrics = run_evaluation(model, Path(__file__).resolve().parent.parent / "output/05_normalized", split='test')
    print(metrics)

    # Plot ADE(t)
    fig, ax = plot_horizon_error(metrics, label="Constant Velocity")
    plt.show()
