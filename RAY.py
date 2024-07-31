from Model import Model
from flask import Flask, render_template, request, send_from_directory, jsonify
import os



app = Flask(__name__)
app.config["CHATS"] = "./chats"
app.config['UPLOAD_FOLDER'] = './uploads'


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/messages", methods = ["GET", "POST"])
def messages():
    return send_from_directory(app.config["CHATS"],"messages.json")

@app.route("/answer", methods = ["GET", "POST"])
def answer():
    data = request.get_json()
    message = data["message"]
    type  =  data["conversationType"]
    print(message, ", ", type)
    if type == "Chat":
        return m.decode(message), {"Content-Type" : "text/plain"}
    else:
        return m.agent(message), {"Content-Type" : "text/plain"}

@app.route("/upload", methods = ["GET", "POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        print("file saved")
        message =request.form.get('message', '')
        print(message)
        return m.RAG(file.filename,message), {"Content-Type" : "text/plain"}
    
    
    

if __name__ == "__main__":
    m = Model(impl="onnx")
    print("model loaded")
    app.run(debug = False,
            port = 8000)