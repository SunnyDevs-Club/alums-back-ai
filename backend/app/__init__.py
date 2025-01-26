
from flask import Flask, request
from sqlalchemy import text

import os

from config import Config
from app.extensions import db, swagger
from app.models import User, Parcel


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Initialize Flask extensions here
    db.init_app(app)
    swagger.init_app(app)

    from app.users.routes import bp as user_bp
    app.register_blueprint(user_bp, url_prefix="/users")

    from app.groups.routes import bp as group_bp
    app.register_blueprint(group_bp, url_prefix="/groups")

    from app.parcels.routes import bp as parcels_bp
    app.register_blueprint(parcels_bp, url_prefix="/parcels")

    from app.tasks.routes import bp as tasks_bp
    app.register_blueprint(tasks_bp, url_prefix="/tasks")

    @app.post('/login')
    def login():
        """Login endpoint.
        method: POST,
        body: {
            user_id: int,
            password: str,
        }

        response:
        {
            "status": bool,
            "role": str
        }
        """

        try:
            data = request.get_json()
            user_id = data['user_id']
            password = data['password']

            user = User.get(user_id)

            if not user or not user.check_password(password):
                return {
                    "status": False,
                    "role": None
                }, 400

            return {
                "status": True,
                "role": user.role
            }, 200

        except Exception as e:
            return {
                "status": False,
                "role": None
            }

    return app
