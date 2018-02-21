from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import pyLDAvis.gensim
from gensim.models import LdaModel, Phrases
from gensim.corpora import Dictionary
from warnings import filterwarnings
filterwarnings(action='ignore', category=UserWarning, module='gensim')
from datetime import datetime

app = Flask(__name__)
cors = CORS(app)


def update_progress(ip: str, percentage: float):
    now = str(datetime.now().strftime("%Y-%m-%d %H:%M"))
    filename = "update_log.json"
    with open(filename) as progress:
        logs = json.load(progress)

    if ip not in logs:
        logs[ip] = {
            'update': {
                'datetime': now,
                'percentage': percentage
            }
        }
    else:
        logs[ip]["update"] = {
            'datetime': now,
            'percentage': percentage
        }

    with open(filename, 'w') as progress:
        json.dump(logs, progress)


def reset_logs(ip: str):
    try:
        filename = "update_log.json"
        with open(filename) as progress:
            update_logs = json.load(progress)
        update_logs[ip]['update'] = {
            'datetime': "",
            'percentage': 0
        }
        with open(filename, 'w') as progress:
            json.dump(update_logs, progress)
    except Exception:
        pass


@app.route('/')
def return_progress():
    try:
        filename = "update_log.json"
        with open(filename) as progress:
            update_logs = json.load(progress)
        # return str(1)
        logs = update_logs[request.remote_addr]['update']
        return str(logs['percentage'])
    except Exception:
        return str(0)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=63343, debug=True)

