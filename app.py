from base64 import b64encode

from flask import Flask, render_template, request, redirect

from worker import generate_data, get_differences

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/first_task")
def first_task():
    if "file" not in request.files:
        return redirect("index")
    file = request.files["file"]
    if not file or not file.filename:
        return redirect("index")
    output = generate_data(file.stream.read())
    output_images = [(b64encode(first_image).decode(), b64encode(second_image).decode())
                     for first_image, second_image in output]
    return render_template("index.html", output_images=output_images)


@app.post("/second_task")
def second_task():
    if "first_file" not in request.files or "second_file" not in request.files:
        return redirect("index")
    first_file = request.files["first_file"]
    second_file = request.files["second_file"]
    if not first_file or not first_file.filename or not second_file or not second_file.filename:
        return redirect("index")
    first_image, second_image = get_differences(first_file.stream.read(), second_file.stream.read())
    output_images = [(b64encode(first_image).decode(), b64encode(second_image).decode())]
    return render_template("index.html", output_images=output_images)


if __name__ == "__main__":
    app.run(debug=True)
