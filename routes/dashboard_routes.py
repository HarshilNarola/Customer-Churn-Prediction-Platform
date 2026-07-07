from flask import Blueprint
from flask import render_template


dashboard = Blueprint(
    "dashboard",
    __name__
)


@dashboard.route("/dashboard")
def dashboard_page():

    images = [

        # Target
        "target_distribution.png",
        "target_percentage.png",
        "correlation_heatmap.png",

        # Age
        "age_distribution.png",
        "age_boxplot.png",
        "age_vs_churn_boxplot.png",

        # Gender
        "gender_countplot.png",
        "gender_vs_churn.png",

        # Contract
        "contract_length_countplot.png",
        "contract_length_vs_churn.png",

        # Subscription
        "subscription_type_countplot.png",
        "subscription_type_vs_churn.png",

        # Tenure
        "tenure_distribution.png",
        "tenure_boxplot.png",
        "tenure_vs_churn_boxplot.png",

        # Usage
        "usage_frequency_distribution.png",
        "usage_frequency_boxplot.png",
        "usage_frequency_vs_churn_boxplot.png",

        # Support Calls
        "support_calls_distribution.png",
        "support_calls_boxplot.png",
        "support_calls_vs_churn_boxplot.png",

        # Payment Delay
        "payment_delay_distribution.png",
        "payment_delay_boxplot.png",
        "payment_delay_vs_churn_boxplot.png",

        # Total Spend
        "total_spend_distribution.png",
        "total_spend_boxplot.png",
        "total_spend_vs_churn_boxplot.png",

        # Last Interaction
        "last_interaction_distribution.png",
        "last_interaction_boxplot.png",
        "last_interaction_vs_churn_boxplot.png"

    ]

    return render_template(
        "dashboard.html",
        images=images
    )