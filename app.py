from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import Samuel

app = Flask(__name__)
cors = CORS(app)


def valid(api_key):
    return requests.get("http://192.168.1.14/validate_key?key="+api_key).json()


@app.route('/')
def get_tasks():
    return "Welcome to SAMUEL API"


@app.route('/samuel_api', methods=['POST', 'GET'])
def samuel_api():
    if valid(request.args.get('KEY')):
        # try:
        samuel = Samuel.api(request.get_json())
        return jsonify(samuel)
        # except Exception as ex:
        #     print("SYSTEM ERROR")
        #     return "SYSTEM ERROR"
    else:
        return "Invalid API Key"


@app.route('/samuel_init', methods=['GET'])
def samuel_init():
    return jsonify(Samuel.init(request.args.get('KEY')))


@app.route('/samuel_validate', methods=['GET','POST'])
def samuel_validate():
    if valid(request.args.get('KEY')):
        return "<h1>Valid API Key</h1>"
    else:
        return "<h1>Invalid API Key</h1>"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=63342, debug=True)
