import threading
from game_model.game import Game
from flask import Flask, Response, jsonify, abort
from flask_cors import CORS
import json


from game_model.turn import Turn

class BoundData:
    def __init__(self):
        self.game : Game
        self.game_json_memory: list[str] = []
        self.lock_object: threading.Lock = threading.Lock()
    
    def on_next_game_state(self, game: Game, turn: Turn):
        self.game = game
        self.game_json_memory.append(json.dumps(game_data.game.as_serializable_data()))

game_data: BoundData = BoundData()

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "Welcome to the index"

@app.route("/game/json")
def get_game_json():
    acquired = game_data.lock_object.acquire(timeout=5)
    if not acquired:
        return {
            "Error": "could not get lock on game object after 5 seconds"
        }
    try:
        json_data = jsonify(game_data.game.as_serializable_data())
    finally:
        game_data.lock_object.release()

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
    return app.response_class(
        response=game_data.game_json_memory[game_id],
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