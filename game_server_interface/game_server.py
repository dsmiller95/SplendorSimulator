import threading
from game_model.game import Game
from flask import Flask, Response, jsonify, abort
from flask_cors import CORS
import json


from game_model.turn import Turn

## ~ apprx 2500 bytes per game mem. this is 250 megabytes.
## only this # will be a valid range to query from the api.
## there is a better way to do this, probably.
max_game_memory_size = 100000 

class BoundData:
    def __init__(self):
        self.game : Game
        self.game_json_memory: list[str] = []
    
    def on_next_game_state(self, game: Game, turn: Turn):
        self.game = game
        self.game_json_memory.append(json.dumps(game_data.game.as_serializable_data()))
        if len(self.game_json_memory) > 100000:
            self.game_json_memory[-max_game_memory_size] = None



game_data: BoundData = BoundData()

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "Welcome to the index"

@app.route("/game/json")
def get_game_json():
    json_data = jsonify(game_data.game.as_serializable_data())

    return json_data

@app.route("/history/length")
def get_history_length():

    return jsonify(
        length= len(game_data.game_json_memory)
    )

@app.route("/history/game/<game_id>")
def get_historical_game(game_id=0):
    game_id = int(game_id)
    if game_id >= len(game_data.game_json_memory) or game_id < 0:
        abort(404)
    response_data = game_data.game_json_memory[game_id]
    if response_data is None:
        abort(404)
    return app.response_class(
        response=response_data,
        status=200,
        mimetype="application/json"
    )


@app.route("/game/text")
def get_game_description():
    return f"""
<html>
	<head><meta name="color-scheme" content="light dark"></head>
	<body>
		<pre style="word-wrap: break-word; white-space: pre-wrap;">{game_data.game.describe_common_state()}</pre>
	</body>
</html>"""