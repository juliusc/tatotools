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

os.makedirs("data/base", exist_ok=True)
os.makedirs("data/warehouse", exist_ok=True)

class NotATSError(Exception):
    pass

# Store timestamp in session to persist across pages
@app.route('/')
def index():
    return redirect("/plan")

# @app.route('/assets')
# def assets():
#     return render_template('assets.html')

@app.route('/apikey')
def apikey():
    return render_template('apikey.html')

@app.route('/plan')
def plan():
    return render_template('plan.html')

@app.route('/plan-old')
def plan_old():
    return render_template('plan-old.html')

@app.route('/fetch-bases', methods=['GET'])
def fetch_bases():
    api_key = request.headers.get("Gt-Api-Key")
    print(api_key)
    bases_data = lib.load_bases_data(api_key=api_key)
    return jsonify(sorted([(base_data["name"], base_data["id"]) for base_data in bases_data.values()])) 


@app.route('/compute-plan')
def compute_plan():
    api_key = request.headers.get("Gt-Api-Key")
    company_data, bases_data = lib.load_company_and_bases_data(api_key)
    if company_data["gId"] != 39:
        return "You must be a member of ATS", 401
    exchange_data = lib.load_exchange_data(api_key)
    exchange_orders_data = lib.load_exchange_orders(api_key)

    base_ids = [int(x) for x in request.args.getlist("base_ids")]
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

    warehouses_data = lib.load_warehouses_data(api_key)
    plan_orders = optimization.plan_production2(
        base_ids,
        company_data,
        bases_data,
        warehouses_data,
        budget,
        duration,
        exchange_data,
        exchange_orders_data,
        consumables_duration=consumables_duration,
        max_weight=max_weight,
        include_consumables=include_consumables,
        include_exchange=include_exchange,
        only_outputs_sellable=only_outputs_sellable)

    return optimization.prepare_report(
        base_ids,
        plan_orders,
        consumables_duration or duration,
        company_data,
        bases_data,
        warehouses_data)


@app.route('/compute-plan-old')
def compute_plan_old():
    api_key = request.headers.get("Gt-Api-Key")
    company_data, bases_data = lib.load_company_and_bases_data(api_key)
    exchange_data = lib.load_exchange_data(api_key)
    exchange_orders_data = lib.load_exchange_orders(api_key)

    base_ids = [int(x) for x in request.args.getlist("base_ids")]
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

    warehouses_data = lib.load_warehouses_data(api_key)
    purchases_df, order_duration_df, orders_df, mats_df, wishlist = optimization.plan_production(
        base_ids[0],
        company_data,
        bases_data,
        warehouses_data,
        budget,
        duration,
        exchange_data,
        exchange_orders_data,
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
    api_key = request.headers.get("Gt-Api-Key")
    headers = {
        'accept': 'text/plain',
        'Content-Type': 'application/json'
    }

    base_ids = [x["wishlistId"] for x in request.json]
    # print("deleting...")
    # r1 = requests.post(f"{API_BASE_URL}/wishlists/deletemany?apikey={api_key}", json=base_ids)
    # print("deletemany:", r1.status_code, r1.text)

    print("uploading...")
    r2 = requests.post(f"{API_BASE_URL}/wishlists/additems?apikey={api_key}", json=request.json)
    print("additems:", r2.status_code, r2.text)
    if r2.status_code == 429:
        return "Rate limit exceeded", 429


    breakpoint()
    print("done")

    return {"status": "ok"}


@app.route('/fetch-supplies', methods=['GET'])
def compute_supplies():
    api_key = request.headers.get("Gt-Api-Key")
    # lib.fetch_api_data_if_needed(f"{API_BASE_URL}/company?apikey={api_key}", "data/company_data.json", 180)
    company_data, bases_data = lib.load_company_and_bases_data(api_key=api_key)
    warehouses_data = lib.load_warehouses_data(api_key=api_key)

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

        prod_time_max = optimization.get_full_production_time(base_id, bases_data, company_data, warehouses_data, limit_by_prod_orders=False)
        prod_time_actual = optimization.get_full_production_time(base_id, bases_data, company_data, warehouses_data, limit_by_prod_orders=True)

        rows.append({
            "name": base_data["name"],
            "base_id": base_data["id"],
            "hrs_until_slowdown": lib.get_time_string(min(min_consum_time, prod_time_actual)),
            "consumables_hrs": lib.get_time_string(min_consum_time),
            "prod_hrs_max": lib.get_time_string(prod_time_max),
            "prod_hrs_actual": lib.get_time_string(prod_time_actual),
            "key": min(min_consum_time, prod_time_actual)
        })

    rows = sorted(rows, key=lambda x: x["key"])

    return jsonify(rows)