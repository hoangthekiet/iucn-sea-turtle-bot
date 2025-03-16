import os
import requests
try:
    from flask import jsonify
except ImportError as e:
    print(e)


class BaseService:
    __root_folder = os.getenv("UPLOAD_FOLDER", "/tmp")

    @classmethod
    def handle(cls):
        pass

    @classmethod
    def build_output(cls, msg):
        output = {"data": msg}
        return jsonify(output)

    @classmethod
    def post_requests(cls, url, data_json, headers=None):
        response = requests.post(
            url,
            headers=headers,
            json=data_json
        )
        if response.status_code != 200:
            raise Exception(response.text)
        body_data = response.json().get("data", {})
        return body_data
