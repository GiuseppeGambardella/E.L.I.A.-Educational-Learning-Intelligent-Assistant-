from flask import Flask
from elia.config import Config

def create_app():
    app = Flask(__name__, static_folder="server/static", static_url_path="/static")
    app.config.from_object(Config)

    # Blueprints
    from .routes.health import bp as health_bp
    app.register_blueprint(health_bp, url_prefix="/_")
    return app
