from game_model.game import Game
from flask import Flask, Response, jsonify

class BoundData:
    def __init__(self):
        self.game : Game

game_data = BoundData()

app = Flask(__name__)

@app.route("/")
def index():
    return "Welcome to the index"

@app.route("/game/json")
def get_game_json():
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