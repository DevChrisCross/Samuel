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
    processed = Samuel.process(data['corpus'], data['summary_length'],data['query'])
    return jsonify({
        'summarized_text': processed['summarized_text'],
        'polarity': processed['polarity'],
    })


@app.route('/data_entry', methods=['POST'])
def data_entry():
    data = request.get_json()
    return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
