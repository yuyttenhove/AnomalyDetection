from flask import Flask
from flask_restful import Api, Resource, reqparse
from anomaly import score

app = Flask(__name__)
api = Api(app)

datapoint = {"x":2, "y":3}

class Datapoint(Resource):
    def get(self):
        if datapoint is not None:
            return datapoint, 200
        return "No datapoint submitted", 404
    
    def put(self):
        global datapoint
        parser = reqparse.RequestParser()
        parser.add_argument("x")
        parser.add_argument("y")
        args = parser.parse_args()

        if datapoint is not None:
            datapoint["x"] = args["x"]
            datapoint["y"] = args["y"]
            return score(datapoint), 200
        
        datapoint = {"x":args["x"], "y":args["y"]}
        return score(datapoint), 201
    
    def delete(self):
        global datapoint
        datapoint = None
        return "Deleted datapoint.", 200

api.add_resource(Datapoint, "/datapoint")
app.run(debug=True, host="0.0.0.0", port="5000", use_reloader=False)