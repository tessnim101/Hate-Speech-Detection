import pandas as pd
import matplotlib.pyplot as plt

RUN_ID = "2026-04-30_08-57-41"

def load_and_prepare(path, run_id):
    df = pd.read_csv(path)

    df = df[df["run_id"] == run_id].copy()

    train_df = df[df["loss"].notna()].copy()
    eval_df = df[df["eval_loss"].notna()].copy()

    train_df = train_df[["epoch", "loss"]]
    eval_df = eval_df[
        [
            "epoch",
            "eval_loss",
            "eval_accuracy",
            "eval_f1_macro",
            "eval_f1_binary",
            "eval_f1_class0",
            "eval_f1_class1",
        ]
    ]

    merged = pd.merge(train_df, eval_df, on="epoch", how="inner")
    merged = merged.sort_values("epoch").rename(columns={"loss": "train_loss"})

    return merged


def plot_losses(df):
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    plt.plot(df["epoch"], df["eval_loss"], marker="o", label="Validation Loss")

    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_metrics(df):
    plt.figure()
    plt.plot(df["epoch"], df["eval_accuracy"], marker="o", label="Accuracy")
    plt.plot(df["epoch"], df["eval_f1_macro"], marker="o", label="F1 Macro")

    plt.title("Validation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_per_class_f1(df):
    plt.figure()
    plt.plot(df["epoch"], df["eval_f1_class0"], marker="o", label="Class 0")
    plt.plot(df["epoch"], df["eval_f1_class1"], marker="o", label="Class 1")

    plt.title("F1 Score per Class")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_summary(df):
    best_row = df.loc[df["eval_f1_macro"].idxmax()]

    print("\n===== SUMMARY =====")
    print(f"Best epoch: {best_row['epoch']}")
    print(f"Best F1_macro: {best_row['eval_f1_macro']:.4f}")
    print(f"Accuracy: {best_row['eval_accuracy']:.4f}")
    print(f"F1 class 0: {best_row['eval_f1_class0']:.4f}")
    print(f"F1 class 1: {best_row['eval_f1_class1']:.4f}")


if __name__ == "__main__":
    path = "results/results.csv"
    run_id = RUN_ID

    df = load_and_prepare(path, run_id)

    print(df)
    print_summary(df)

    plot_losses(df)
    plot_metrics(df)
    plot_per_class_f1(df)