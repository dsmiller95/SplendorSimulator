import threading
from game_model.game import Game
from flask import Flask, Response, jsonify

class BoundData:
    def __init__(self):
        self.game : Game
        self.lock_object: threading.Lock = threading.Lock()

game_data: BoundData = BoundData()

app = Flask(__name__)

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

    return game_data.game.as_serializable_data()
    ## return Response(game_data.game.as_serializable_data(), mimetype='text/json')

@app.route("/game/text")
def get_game_description():
    return f"""
<html>
	<head><meta name="color-scheme" content="light dark"></head>
	<body>
		<pre style="word-wrap: break-word; white-space: pre-wrap;">{game_data.game.describe_common_state()}</pre>
	</body>
</html>"""