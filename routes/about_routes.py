from flask import Blueprint
from flask import render_template


about = Blueprint(
    "about",
    __name__
)


@about.route("/about")
def about_page():

    return render_template(
        "about.html"
    )