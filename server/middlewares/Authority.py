from functools import wraps
from flask import request
from flask import current_app as app



def api_key_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "authorization" in request.headers:
            token = request.headers.get("authorization", None)
        elif "token" in request.args:
            token = request.args.get("token", None)
        else:
            return "unauthorized", 401
        
        if token == app.config["SECRET_KEY"]:
            return fn(*args, **kwargs)
        return "unauthorized", 401

    return wrapper

def no_authen(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper
