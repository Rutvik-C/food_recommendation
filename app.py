from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pickle
import pandas as pd


app = Flask(__name__)
api = Api(app)

model = pickle.load(open("./model/food_rec.pkl", "rb"))

df = pd.read_csv("./dataset/food_recommendation.csv")
names = df["name"].copy()
features = df.drop(["name"], axis=1)


class MachineLearning(Resource):
    def get(self, food_index):
        try:
            test = features.iloc[food_index]

            rec = list()
            dist, ind = model.kneighbors([test], n_neighbors=6)
            for i in ind[0][1:]:
                rec.append(names[i])

            return jsonify({"result": "success", "recommendation": rec})

        except Exception as e:
            return jsonify({"result": "failed", "error_message": e})


api.add_resource(MachineLearning, "/ml/<int:food_index>")

if __name__ == "__main__":
    app.run(debug=True)

