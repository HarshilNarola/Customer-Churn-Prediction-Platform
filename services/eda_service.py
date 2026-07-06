import os

import matplotlib.pyplot as plt
import pandas as pd


class EDAService:
    """
    Provides visualization utilities for Exploratory Data Analysis.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.output_dir = "static/images/eda"

        os.makedirs(self.output_dir, exist_ok=True)

    # ==========================================================
    # Private Helper Function
    # ==========================================================

    def _finalize_plot(self, filename: str, save: bool, show: bool):
        """
        Saves and/or displays the current plot.
        """

        if save:
            plt.savefig(
                os.path.join(self.output_dir, filename),
                dpi=300,
                bbox_inches="tight",
            )

        if show:
            plt.show()

        plt.close()

    # ==========================================================
    # Target Variable Analysis
    # ==========================================================

    def plot_target_distribution(self, save=True, show=True):

        fig, ax = plt.subplots(figsize=(7, 5))

        self.df["Churn"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Customer Churn Distribution")
        ax.set_xlabel("Churn")
        ax.set_ylabel("Number of Customers")

        self._finalize_plot(
            "target_distribution.png",
            save,
            show,
        )

    def plot_target_percentage(self, save=True, show=True):

        fig, ax = plt.subplots(figsize=(6, 6))

        self.df["Churn"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax
        )

        ax.set_ylabel("")
        ax.set_title("Customer Churn Percentage")

        self._finalize_plot(
            "target_percentage.png",
            save,
            show,
        )

    # ==========================================================
    # Numerical Feature Analysis
    # ==========================================================

    def plot_histogram(
        self,
        column_name,
        bins=30,
        save=True,
        show=True,
    ):

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(
            self.df[column_name],
            bins=bins
        )

        ax.set_title(f"{column_name} Distribution")
        ax.set_xlabel(column_name)
        ax.set_ylabel("Frequency")

        filename = (
            column_name.lower()
            .replace(" ", "_")
            + "_distribution.png"
        )

        self._finalize_plot(
            filename,
            save,
            show,
        )

    def plot_boxplot(
        self,
        column_name,
        save=True,
        show=True,
    ):

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.boxplot(self.df[column_name])

        ax.set_title(f"{column_name} Boxplot")
        ax.set_ylabel(column_name)

        filename = (
            column_name.lower()
            .replace(" ", "_")
            + "_boxplot.png"
        )

        self._finalize_plot(
            filename,
            save,
            show,
        )

    # ==========================================================
    # Categorical Feature Analysis
    # ==========================================================

    def plot_countplot(
        self,
        column_name,
        save=True,
        show=True,
    ):

        fig, ax = plt.subplots(figsize=(8, 5))

        self.df[column_name].value_counts().plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(f"{column_name} Distribution")
        ax.set_xlabel(column_name)
        ax.set_ylabel("Count")

        filename = (
            column_name.lower()
            .replace(" ", "_")
            + "_countplot.png"
        )

        self._finalize_plot(
            filename,
            save,
            show,
        )

    def plot_countplot_by_target(
        self,
        column_name,
        save=True,
        show=True,
    ):

        fig, ax = plt.subplots(figsize=(8, 5))

        pd.crosstab(
            self.df[column_name],
            self.df["Churn"]
        ).plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(f"{column_name} vs Churn")
        ax.set_xlabel(column_name)
        ax.set_ylabel("Count")

        filename = (
            column_name.lower()
            .replace(" ", "_")
            + "_vs_churn.png"
        )

        self._finalize_plot(
            filename,
            save,
            show,
        )

    # ==========================================================
    # Numerical Feature vs Target
    # ==========================================================

    def plot_boxplot_by_target(
        self,
        column_name,
        target="Churn",
        save=True,
        show=True,
    ):

        fig, ax = plt.subplots(figsize=(8, 5))

        self.df.boxplot(
            column=column_name,
            by=target,
            ax=ax,
        )

        ax.set_title(f"{column_name} vs {target}")
        ax.set_xlabel(target)
        ax.set_ylabel(column_name)

        # Remove pandas default title
        plt.suptitle("")

        filename = (
            column_name.lower()
            .replace(" ", "_")
            + "_vs_churn_boxplot.png"
        )

        self._finalize_plot(
            filename,
            save,
            show,
        )

    def plot_correlation_heatmap(
        self,
        save=True,
        show=True,
    ):

        fig, ax = plt.subplots(figsize=(10, 8))

        correlation = self.df.drop(columns=["CustomerID"]).corr(numeric_only=True)

        image = ax.imshow(
            correlation,
            cmap="coolwarm",
            aspect="auto",
        )

        ax.set_xticks(range(len(correlation.columns)))
        ax.set_yticks(range(len(correlation.columns)))

        ax.set_xticklabels(
            correlation.columns,
            rotation=45,
            ha="right",
        )

        ax.set_yticklabels(
            correlation.columns
        )

        plt.colorbar(image)

        ax.set_title("Correlation Heatmap")

        plt.tight_layout()

        self._finalize_plot(
            "correlation_heatmap.png",
            save,
            show,
        )