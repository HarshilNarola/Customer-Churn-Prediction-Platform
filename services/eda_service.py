from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class EDAService:
    """
    Provides visualization utilities for
    Exploratory Data Analysis (EDA).
    """

    OUTPUT_DIRECTORY = Path("static/images/eda")

    def __init__(
        self,
        dataframe: pd.DataFrame
    ) -> None:
        """
        Initialize the EDA service.
        """

        self.dataframe = dataframe

        self.OUTPUT_DIRECTORY.mkdir(

            parents=True,

            exist_ok=True

        )

    # ==========================================================
    # Private Helper Methods
    # ==========================================================

    def _generate_filename(
        self,
        name: str,
        suffix: str
    ) -> str:
        """
        Generate a standardized filename.
        """

        return (

            name.lower()

            .replace(" ", "_")

            + suffix

        )

    def _finalize_plot(
        self,
        filename: str,
        save: bool = True,
        show: bool = True
    ) -> None:
        """
        Save and/or display the current plot.
        """

        if save:

            plt.savefig(

                self.OUTPUT_DIRECTORY / filename,

                dpi=300,

                bbox_inches="tight"

            )

        if show:

            plt.show()

        plt.close()

    # ==========================================================
    # Target Variable Analysis
    # ==========================================================

    def plot_target_distribution(
        self,
        save: bool = True,
        show: bool = True
    ) -> None:

        fig, ax = plt.subplots(figsize=(7, 5))

        self.dataframe["Churn"].value_counts().plot(

            kind="bar",

            ax=ax

        )

        ax.set_title("Customer Churn Distribution")

        ax.set_xlabel("Churn")

        ax.set_ylabel("Number of Customers")

        self._finalize_plot(

            "target_distribution.png",

            save,

            show

        )

    def plot_target_percentage(
        self,
        save: bool = True,
        show: bool = True
    ) -> None:

        fig, ax = plt.subplots(figsize=(6, 6))

        self.dataframe["Churn"].value_counts().plot(

            kind="pie",

            autopct="%1.1f%%",

            ax=ax

        )

        ax.set_ylabel("")

        ax.set_title("Customer Churn Percentage")

        self._finalize_plot(

            "target_percentage.png",

            save,

            show

        )

    # ==========================================================
    # Numerical Feature Analysis
    # ==========================================================

    def plot_histogram(
        self,
        column_name: str,
        bins: int = 30,
        save: bool = True,
        show: bool = True
    ) -> None:

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(

            self.dataframe[column_name],

            bins=bins

        )

        ax.set_title(f"{column_name} Distribution")

        ax.set_xlabel(column_name)

        ax.set_ylabel("Frequency")

        self._finalize_plot(

            self._generate_filename(

                column_name,

                "_distribution.png"

            ),

            save,

            show

        )

    def plot_boxplot(
        self,
        column_name: str,
        save: bool = True,
        show: bool = True
    ) -> None:

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.boxplot(

            self.dataframe[column_name]

        )

        ax.set_title(f"{column_name} Boxplot")

        ax.set_ylabel(column_name)

        self._finalize_plot(

            self._generate_filename(

                column_name,

                "_boxplot.png"

            ),

            save,

            show

        )

    # ==========================================================
    # Categorical Feature Analysis
    # ==========================================================

    def plot_countplot(
        self,
        column_name: str,
        save: bool = True,
        show: bool = True
    ) -> None:

        fig, ax = plt.subplots(figsize=(8, 5))

        self.dataframe[column_name].value_counts().plot(

            kind="bar",

            ax=ax

        )

        ax.set_title(f"{column_name} Distribution")

        ax.set_xlabel(column_name)

        ax.set_ylabel("Count")

        self._finalize_plot(

            self._generate_filename(

                column_name,

                "_countplot.png"

            ),

            save,

            show

        )

    def plot_countplot_by_target(
        self,
        column_name: str,
        save: bool = True,
        show: bool = True
    ) -> None:

        fig, ax = plt.subplots(figsize=(8, 5))

        pd.crosstab(

            self.dataframe[column_name],

            self.dataframe["Churn"]

        ).plot(

            kind="bar",

            ax=ax

        )

        ax.set_title(f"{column_name} vs Churn")

        ax.set_xlabel(column_name)

        ax.set_ylabel("Count")

        self._finalize_plot(

            self._generate_filename(

                column_name,

                "_vs_churn.png"

            ),

            save,

            show

        )

    # ==========================================================
    # Numerical Feature vs Target
    # ==========================================================

    def plot_boxplot_by_target(
        self,
        column_name: str,
        target: str = "Churn",
        save: bool = True,
        show: bool = True
    ) -> None:

        fig, ax = plt.subplots(figsize=(8, 5))

        self.dataframe.boxplot(

            column=column_name,

            by=target,

            ax=ax

        )

        plt.suptitle("")

        ax.set_title(f"{column_name} vs {target}")

        ax.set_xlabel(target)

        ax.set_ylabel(column_name)

        self._finalize_plot(

            self._generate_filename(

                column_name,

                "_vs_churn_boxplot.png"

            ),

            save,

            show

        )

    # ==========================================================
    # Correlation Heatmap
    # ==========================================================

    def plot_correlation_heatmap(
        self,
        save: bool = True,
        show: bool = True
    ) -> None:

        fig, ax = plt.subplots(figsize=(10, 8))

        correlation = self.dataframe.drop(

            columns=["CustomerID"],

            errors="ignore"

        ).corr(

            numeric_only=True

        )

        image = ax.imshow(

            correlation,

            cmap="coolwarm",

            aspect="auto"

        )

        ax.set_xticks(

            range(len(correlation.columns))

        )

        ax.set_yticks(

            range(len(correlation.columns))

        )

        ax.set_xticklabels(

            correlation.columns,

            rotation=45,

            ha="right"

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

            show

        )