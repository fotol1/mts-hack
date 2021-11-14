from typing import Optional
import json

from fastapi import FastAPI
import numpy as np
import torch

from models import multivae

app = FastAPI()

model = multivae.MultiVAE([200, 600, 50], dropout=0.0)
model.load_state_dict(torch.load("../Data/artifacts/multvae_mcc"))
model.eval()

with open("../Data/artifacts/params_merchant.json", "r") as f:
    params = json.load(f)


model_m = multivae.MultiVAE([200, 600, params["n_items"]], dropout=0.0)
model_m.load_state_dict(torch.load("../Data/artifacts/multvae_merchant"))
model_m.eval()


with open("../Data/artifacts/item2idx_mcc.json", "r") as f:
    item2idx = json.load(f)

with open("../Data/artifacts/item2idx_merchant.json", "r") as f:
    item2idx_merchant = json.load(f)

# item2idx_merchant = {str(x): v for x, v in item2idx_merchant.items()}

decoder = {
    "5411": " Grocery Stores,supermarkets",
    "7399": " Business Services",
    "5814": " Fast Food Res.",
    "5921": " Package Stores - Beer, Liqu",
    "9402": " Postal Services-Government ",
    "4121": " Taxicabs/limousines",
    "5499": " Miscellaneous Food Stores",
    "4131": " Bus Lines",
    "5651": " Family Clothing Stores",
    "5691": " Men's And Ladies's Clothing",
    "5912": " Drug Stores,pharmacies",
    "5964": " Direct Marketing-Catalog Me",
    "5722": " Household Appliance Stores",
    "4111": " Local/Suburban Commuter Pas",
    "5943": " Stationery,office,and Schoo",
    "5311": " Department stores",
    "5977": " Cosmetic Stores",
    "5200": " Home Supply,Warehouse Store",
    "7230": " Beauty Shops & Barber Shops",
    "4814": " Telecommunication Service",
    "5999": " Miscellaneous & specialty r",
    "5641": " Children's And Infant's Wea",
    "4816": " Computer Network/Informatio",
    "5945": " Hobby,toy,and Game Shops",
    "5541": " Service Stations",
    "5399": " Miscellaneous General Merch",
    "5942": " Book Stores",
    "5451": " Dairy Products Stores",
    "5331": " Variety Stores",
    "5732": " Electronic Sales",
    "5533": " Automotive Parts,acces. Sto",
    "7995": " Casino",
    "5661": " Shoe stores",
    "7311": " Advertising Services",
    "5812": " Eating Places,Restaurants",
    "5941": " Sporting Goods Stores",
    "5813": " Drinking Places - Bars,Tave",
    "4112": " Passenger RailwaysX",
    "5211": " Lumber And Building Materia",
    "5441": " Candy,nut,confectionary Sto",
    "4812": " Telephone Service/Equip....",
    "5992": " Florists",
    "5462": " Bakeries",
    "5995": " Pet Shops-pet Foods & Suppl",
    "4900": " Utilities-electric,gas,wate",
    "5993": " Cigar Stores And Stands",
    "7994": " Video Games Arcades/establi",
}


idx2item = {v: k for k, v in item2idx.items()}
idx2item_merchant = {v: k for k, v in item2idx_merchant.items()}


# torch.load()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/category_recommendations/")
def read_item(q: Optional[str] = None):

    recs = get_recs(model, interactions=q, item2idx=item2idx, idx2item=idx2item)

    answer = {"user_mcc": q}
    answer.update(recs)
    print(recs)

    return answer


@app.get("/merchant_recommendations/")
def read_item(q: Optional[str] = None):

    recs = get_recs(
        model_m,
        interactions=q,
        item2idx=item2idx_merchant,
        idx2item=idx2item_merchant,
        no_names=True,
        num_items=params["n_items"],
    )

    answer = {"user_interactions": q}
    answer.update(recs)
    print(recs)

    return answer


def get_recs(model, interactions, item2idx, idx2item, num_items=50, topk=10, no_names=False):
    vector = np.zeros(num_items)
    for el in interactions.split("__"):
        # print(el)
        if num_items == 50:
            vector[item2idx.get(str(el), 0)] = 1
        else:
            vector[int(el)] = 1

    vector = torch.Tensor([vector])
    x_recon, _, _ = model(vector)
    preds = x_recon.detach().cpu().numpy()
    preds = np.argsort(-preds)[0]

    preds_decoded = [idx2item[x] for x in preds]
    new_recs = [x for x in preds_decoded if x not in interactions][:topk]
    visited_recs = [x for x in preds_decoded if x in interactions][:topk]

    if no_names:
        return {"merchants_id_recommended": ";".join(map(str, preds_decoded[:topk]))}

    preds_names = [idx2item[x] for x in preds if x in idx2item and idx2item[x] in decoder.keys()]

    new_recs_names = [decoder[x] for x in preds_names if x not in interactions][:topk]
    visited_recs_names = [decoder[x] for x in preds_names if x in interactions][:topk]
    preds_names = [decoder[x] for x in preds_names]

    return {
        "new_categories_id": ";".join(map(str, new_recs)),
        "all_categories_id": ";".join(map(str, preds_decoded[:topk])),
        "visited_categories_id": ";".join(map(str, visited_recs[:topk])),
        "all_categories_names": ";".join(map(str, preds_names[:topk])),
        "new_categories_names": ";".join(map(str, new_recs_names[:topk])),
        "visited_categories_names": ";".join(map(str, visited_recs_names[:topk])),
    }


if __name__ == "__main__":
    print(1)
