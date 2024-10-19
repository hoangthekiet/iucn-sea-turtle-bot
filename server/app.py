from flask import Flask
from server.extensions import health
from server.config import app_config
from flask_cors import CORS
from server.controllers import ChatController


def create_app(env_name):
    # Create Flask server load server.config
    app = Flask(__name__)
    CORS(app)
    app.config.from_object(app_config[env_name])
    # jwt.init_app(app)
    service_name = app.config['SERVICE_NAME']
    base_prefix = '/' + service_name

    # route
    app.register_blueprint(ChatController.my_service_api, url_prefix=base_prefix + '/chat-service')

    # healthcheck
    app.add_url_rule(base_prefix + "/healthcheck", "healthcheck", view_func=lambda: health.run())
    app.app_context().push()
    return app
