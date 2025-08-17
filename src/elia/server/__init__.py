from flask import Flask
from elia.config import Config
from elia.server.routes.health import bp as health_bp
from elia.server.routes.ask import bp as transcribe_bp
from elia.server.routes.attention import bp as attention_bp

def create_app():
    app = Flask(__name__, static_folder="server/static", static_url_path="/static")
    app.config.from_object(Config)

    # Blueprints
    app.register_blueprint(transcribe_bp, url_prefix="")
    app.register_blueprint(health_bp, url_prefix="")
    app.register_blueprint(attention_bp, url_prefix="")
    return app
