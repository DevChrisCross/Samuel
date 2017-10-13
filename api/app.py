from flask import Flask, jsonify, abort, request
from flask_cors import CORS
from api import Samuel

app = Flask(__name__)
cors = CORS(app)


@app.route('/')
def get_tasks():
    return "Welcome"


@app.route('/samuel_api', methods=['POST'])
def sample_input():
    data = request.get_json()
    summarized_corpus = Samuel.process(data['corpus'], data['summary_length'])
    return jsonify({'summarized_corpus': summarized_corpus})


@app.route('/data_entry', methods=['POST'])
def data_entry():
    data = request.get_json()
    return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
