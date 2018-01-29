from flask import Flask, jsonify, abort, request
from flask_cors import CORS
from api import Samuel

app = Flask(__name__)
cors = CORS(app)


@app.route('/')
def get_tasks():
    return "Welcome to SAMUEL API"


@app.route('/samuel_api', methods=['POST'])
def samuel_api():
    return jsonify(Samuel.api(request.get_json()))


@app.route('/samuel_init', methods=['GET'])
def data_entry():
    return jsonify(Samuel.init(request.args.get('KEY')))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=63342, debug=True)
