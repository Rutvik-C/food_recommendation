from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
import pickle
import pandas as pd
import sklearn

app = Flask(__name__)
api = Api(app)

user_rec_args = reqparse.RequestParser()
user_rec_args.add_argument("user-uid", type=str, required=True)

sim_rec_args = reqparse.RequestParser()
sim_rec_args.add_argument("item-name", type=str, required=True)


model = pickle.load(open("./model/food_recommendation.pkl", "rb"))

df = pd.read_csv("./dataset/food_recommendation.csv")
names = list(df["name"].copy())
features = df.drop(["name"], axis=1)


class UserRecommendation(Resource):
    def get(self):
        args = user_rec_args.parse_args()
        return jsonify({"side": "user", "your-request": args})

        # try:
        #     test = features.iloc[food_index]
        #
        #     rec = list()
        #     dist, ind = model.kneighbors([test], n_neighbors=6)
        #     for i in ind[0][1:]:
        #         rec.append(names[i])
        #
        #     return jsonify({"result": "success", "recommendation": rec})
        #
        # except Exception as e:
        #     return jsonify({"result": "failed", "error_message": e})


class SimilarRecommendation(Resource):
    def get(self):
        try:
            args = sim_rec_args.parse_args()
            item_name = args["item-name"]

            if item_name in names:
                food_index = names.index(item_name)

                user_request = features.iloc[food_index]
                rec = list()
                dist, ind = model.kneighbors([user_request], n_neighbors=6)
                for i in ind[0][1:]:
                    rec.append(names[i])

                return jsonify({"result": "success", "message": rec, "target": item_name})

            else:
                return jsonify({"result": "error", "message": "item not found"})

        except Exception as e:
            return jsonify({"result": "error", "message": e})


api.add_resource(UserRecommendation, "/user-recommendation")
api.add_resource(SimilarRecommendation, "/similar-recommendation")

if __name__ == "__main__":
    app.run(debug=True)

