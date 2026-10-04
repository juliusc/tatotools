import math

from collections import defaultdict

from flask import Flask, render_template, jsonify, redirect, request
import requests
import json
import os

import numpy as np
import cvxpy as cp

import lib
import optimization

app = Flask(__name__)

API_BASE_URL = "https://api.g2.galactictycoons.com/public"
API_KEY = os.environ["GT_API_KEY"]

os.makedirs("data/base", exist_ok=True)
os.makedirs("data/warehouse", exist_ok=True)

# Store timestamp in session to persist across pages
@app.route('/')
def index():
    return redirect("/supplies")

# @app.route('/assets')
# def assets():
#     return render_template('assets.html')

@app.route('/consumables')
def consumables():
    return render_template('consumables.html')

@app.route('/supplies')
def supplies():
    return render_template('supplies.html')

@app.route('/plan')
def plan():
    return render_template('plan.html')

@app.route('/compute-plan')
def compute_plan():
    lib.fetch_api_data_if_needed(f"{API_BASE_URL}/exchange/mat-details", lib.EXCHANGE_DATA_FILENAME)
    exchange_data = lib.load_exchange_data()
    # exchange_warehouse_data = lib.load_warehouses_data()
    # breakpoint()
    
    # # lib.fetch_api_data_if_needed(f"{API_BASE_URL}/company/bases", lib.BASES_FILENAME)
    # bases = lib.load_bases_data()
    # breakpoint()

    base_ids = [int(x) for x in request.args.getlist("id")]
    budget = request.args["budget"]
    budget = float(budget) if budget else None
    duration = float(request.args["duration"]) / 24
    consumables_duration = request.args["consumables_duration"]
    consumables_duration = float(consumables_duration) / 24 if consumables_duration else None
    max_weight = request.args["max_weight"]
    max_weight = int(max_weight) if max_weight else None

    include_consumables = True if request.args.get("include_consumables") == "true" else False
    include_exchange = True if request.args.get("include_exchange") == "true" else False
    only_outputs_sellable = True if request.args.get("only_outputs_sellable") == "true" else False

    bases_data = lib.load_bases_data()
    warehouses_data = lib.load_warehouses_data(min_time=30)
    purchases_df, order_duration_df, orders_df, mats_df, wishlist = optimization.plan_production(
        base_ids[0],
        bases_data,
        warehouses_data,
        budget,
        duration,
        exchange_data,
        consumables_duration=consumables_duration,
        max_weight=max_weight,
        include_consumables=include_consumables,
        include_exchange=include_exchange,
        only_outputs_sellable=only_outputs_sellable)
    return {
        "Purchases": purchases_df.values.tolist(),
        "Duration": order_duration_df.values.tolist(),
        "Orders": orders_df.values.tolist(),
        "Materials": mats_df.values.tolist(),
        "Wishlist": wishlist
    }

@app.post('/wishlist-add')
def wishlist_add():
    # print(request.json)
    headers = {
        'accept': 'text/plain',
        'Content-Type': 'application/json'
    }
    bases_data = lib.load_bases_data()
    planet_id = bases_data[int(request.json["base_id"])]["planetId"]

    result = requests.post(f"{API_BASE_URL}/wishlist/{planet_id}/additems?apikey={API_KEY}", json=request.json["wishlist"])

    return {}

@app.route('/profitability')
def profitability():
    return render_template('profitability.html')

@app.route('/exchange-orders')
def exchange_orders():
    return render_template('exchange.html')

@app.route('/fetch-exchange-orders')
def compute_exchange_orders():
    lib.fetch_api_data_if_needed(f"{API_BASE_URL}/company/exchangeorders?apikey={API_KEY}", lib.EXCHANGE_ORDERS_FILENAME)
    lib.fetch_api_data_if_needed(f"{API_BASE_URL}/exchange/mat-prices?apikey={API_KEY}", lib.PRICES_FILENAME)
    prices_data = lib.load_prices_dict()
    exchange_orders = lib.load_exchange_orders()
    mats_to_lowest_price = {}
    mats_to_total_on_sale = defaultdict(float)
    for data in exchange_orders:
        mats_to_lowest_price[data["matId"]] = min(mats_to_lowest_price.get(data["matId"], math.inf), data["price"])
        mats_to_total_on_sale[data["matId"]] += data["qTot"]
    rows = []
    for mat, lowest_price in sorted(mats_to_lowest_price.items(), key=lambda x: -mats_to_total_on_sale[x[0]] * mats_to_lowest_price[x[0]]):
        exchange_price_color = "red" if prices_data[mat]["currentPrice"] < lowest_price else "black"
        rows.append([
            lib.render_mat_with_icon(lib.get_material_name(mat), link=True),
            mats_to_total_on_sale[mat],
            lib.format_price(lowest_price / 100),
            f'<span style="color: {exchange_price_color};">{lib.format_price(prices_data[mat]["currentPrice"] / 100)}</span>',
            mats_to_total_on_sale[mat] * mats_to_lowest_price[mat] / 100
        ])
    
    return jsonify(rows)

@app.route('/fetch-api', methods=['POST'])
def fetch_api():
    with open(f"data/timestamp", "w") as _:
        pass

    company_data =  lib.fetch_url(f"{API_BASE_URL}/company?apikey={API_KEY}", "data/company_data.json")

    for base_data in company_data["bases"]:
        base_id = base_data["id"]
        warehouse_id = base_data["warehouseId"]
        lib.fetch_url(f"{API_BASE_URL}/company/base/{base_id}?apikey={API_KEY}", f"data/base/{base_id}.json")
        lib.fetch_url(f"{API_BASE_URL}/company/warehouse/{warehouse_id}?apikey={API_KEY}", f"data/warehouse/{warehouse_id}.json")

    lib.fetch_url(f"{API_BASE_URL}/exchange/mat-prices?apikey={API_KEY}", "data/prices.json")

    # Get current timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    
    # Return JSON response
    return jsonify({
        'status': 'success',
        'timestamp': timestamp,
        'message': 'Server action completed'
    })

@app.route('/fetch-bases', methods=['GET'])
def fetch_bases():
    bases_data = lib.load_bases_data(min_time=30)
    return jsonify(sorted([(base_data["name"], base_data["id"]) for base_data in bases_data.values()])) 
    
# @app.route('/compute-consumables', methods=['GET'])
# def compute_consumables():
#     # breakpoint()
#     base_ids = [int(x) for x in request.args.getlist("id")]
#     days = float(request.args["days"])
#     consumed_mats = set()
#     storage_vec = np.zeros(lib.MAT_VECTOR_SIZE)
#     consum_vec = np.zeros(lib.MAT_VECTOR_SIZE)
#     for base_id in base_ids:
#         base_data = lib.load_base_data(base_id)
#         warehouse_data = lib.load_warehouse_data(lib.get_base_warehouse_id(base_data))
#         storage_vec += lib.get_warehouse_contents_vec(warehouse_data)
#         consum_vec += lib.get_total_base_consumption(base_data) * days
#     consumed_mats = lib.get_positive_mat_ids(consum_vec)
#     needed_mats = np.ceil(consum_vec - storage_vec)
#     rows = [
#         {"Material": lib.render_mat_with_icon(lib.get_mat_name(mat)), "Needed": needed_mats[mat]}
#         for mat in lib.get_positive_mat_ids(needed_mats)
#     ]
#     return jsonify(rows)
#     # ids = request.args.getlist('ids')


@app.route('/fetch-supplies', methods=['GET'])
def compute_supplies():
    lib.fetch_api_data_if_needed(f"{API_BASE_URL}/company?apikey={API_KEY}", "data/company_data.json", 180)
    # company_data = lib.load_company_data()
    bases_data = lib.load_bases_data(min_time=30)
    warehouses_data = lib.load_warehouses_data(min_time=30)
    # for base_data in company_data["bases"]:
    #     base_id = base_data["id"]
    #     warehouse_id = base_data["warehouseId"]
    #     lib.fetch_api_data_if_needed(f"{API_BASE_URL}/company/base/{base_id}?apikey={API_KEY}", f"data/base/{base_id}.json", 180)
    #     lib.fetch_api_data_if_needed(f"{API_BASE_URL}/company/warehouse/{warehouse_id}?apikey={API_KEY}", f"data/warehouse/{warehouse_id}.json", 180)
    # lib.fetch_api_data_if_needed(f"{API_BASE_URL}/exchange/mat-prices?apikey={API_KEY}", "data/prices.json", 180)

    rows = []
    for base_id, base_data in bases_data.items():
        if base_data["vacation"]:
            continue
        base_data = bases_data[base_id]
        warehouse_data = warehouses_data[lib.get_base_warehouse_id(base_data)]
        # warehouse_data = lib.load_warehouse_data(lib.get_base_warehouse_id(base_data))
        consum_vec = lib.get_total_base_consumption(base_data)
        storage_vec = lib.get_warehouse_contents_vec(warehouse_data)
        consum_mats = np.where(consum_vec > 0)

        min_consum_time = (storage_vec[consum_mats] / consum_vec[consum_mats]).min()

        # This seem to be incorrectly flipped but it works. Need to fix 
        prod_time_possible = optimization.get_full_production_time(base_id, bases_data, warehouses_data, limit_by_prod_orders=True)
        prod_time_orders = optimization.get_full_production_time(base_id, bases_data, warehouses_data, limit_by_prod_orders=False)

        rows.append({
            "Base": base_data["name"],
            "Consumable hours": lib.get_time_string(min_consum_time),
            "Prod. hours (possible)": lib.get_time_string(prod_time_possible),
            "Prod. hours (orders)": lib.get_time_string(prod_time_orders),
            "key": min(min_consum_time, prod_time_possible)
        })

    rows = sorted(rows, key=lambda x: x["key"])
    for row in rows:
        del row["key"]

    return jsonify(rows)