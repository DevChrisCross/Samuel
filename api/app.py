from flask import Flask, jsonify, abort, request
from flask_cors import CORS
from api import Samuel

app = Flask(__name__)
cors = CORS(app)


@app.route('/')
def get_tasks():
    return "Welcome"


@app.route('/samuel_api', methods=['POST'])
def samuel_api():
    return jsonify(Samuel.api(request.get_json()))


@app.route('/data_entry', methods=['POST'])
def data_entry():
    return jsonify(request.get_json())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
