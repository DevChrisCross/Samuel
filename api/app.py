from flask import Flask, jsonify, abort, request
from flask_cors import CORS
import Normalize

app = Flask(__name__)
cors = CORS(app)


@app.route('/')
def get_tasks():
    return "Welcome"


@app.route('/sample_input', methods=['POST'])
def sample_input():
    data = request.get_json()
    corpus, tokens, sentence_n = Normalize.normalize_corpus(data['corpus'])
    return jsonify({'corpus': corpus, 'tokens': tokens, "sentence_n": sentence_n})


@app.route('/data_entry', methods=['POST'])
def data_entry():
    data = request.get_json()
    return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
