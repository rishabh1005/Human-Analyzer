from flask import Flask,render_template
app=Flask(__name__)
@app.route('/rohit')
def helloworld1():
    return 'HelloWorld rohit'
app.run()       