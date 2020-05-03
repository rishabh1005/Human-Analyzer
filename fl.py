from flask import Flask,render_template
app=Flask(__name__)

@app.route("/")
def hello():
    return render_template('main.html')
@app.route("/rohit")
def hello1():
    return render_template('service.html')
app.run(debug=True)