from flask import Flask, render_template

app = Flask(__name__)


# Define a rota principal (home)
@app.route("/")
def home():
  return render_template("index.html")


if __name__ == "__main__":
  # Roda o servidor localmente
  app.run(debug=True)
