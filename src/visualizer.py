import matplotlib.pyplot as plt
from pathlib import Path

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def plot_actual_vs_predicted(y_test, y_pred):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.7
    )

    # Perfect prediction line
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")

    plt.savefig(
        REPORTS_DIR / "actual_vs_predicted.png"
    )

    plt.close()

    print("Saved: reports/actual_vs_predicted.png")
    

def plot_residuals(y_test, y_pred):

    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.7
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    plt.savefig(
        REPORTS_DIR / "residual_plot.png"
    )

    plt.close()

    print("Saved: reports/residual_plot.png")