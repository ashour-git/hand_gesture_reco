from flask import Flask, render_template
app = Flask(__name__, template_folder='./templates')

@app.route('/')
def index():
    return render_template("templates\index.html")

if __name__ == '__main__':
    app.run(debug=True)