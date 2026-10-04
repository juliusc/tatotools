import json
import os
import requests
import time

import cvxpy as cp
import numpy as np
import pandas as pd


MAT_NAME_TO_ASSET_NAME = {
    "Rations": "BasicRations",
    "Tools": "BasicTools",
    "Exosuit": "BasicExosuit",
    "Bio-Nutrient Blend": "NutrientBlend",
}

BASES_FILENAME = "data/bases.json"
COMPANY_DATA_FILENAME = "data/company_data.json"
WAREHOUSES_FILENAME = "data/warehouses.json"
MAT_DETAILS_FILENAME = "data/mat_details.json"
EXCHANGE_ORDERS_FILENAME = "data/exchange_orders.json"
EXCHANGE_DATA_FILENAME = "data/exchange_data.json"
PRICES_FILENAME = "data/prices.json"

EPSILON = 1e-3

API_BASE_URL = "https://api.g2.galactictycoons.com/public"
API_KEY = os.environ["GT_API_KEY"]

HQ_LEVEL = 0

def fetch_url(url, save_location):
    response = requests.get(url)
    if response.status_code != 200:
        # breakpoint()
        raise Exception
    with open(save_location, "w") as f:
        response_json = response.json()
        f.write(json.dumps(response_json))
    return response_json


def fetch_api_data_if_needed(url, save_location, min_time=300):
    if not os.path.exists(save_location) or time.time() - os.path.getmtime(save_location) > min_time:
        fetch_url(url, save_location)

def load_company_data(min_time=180):
    fetch_api_data_if_needed(f"{API_BASE_URL}/company?apikey={API_KEY}", COMPANY_DATA_FILENAME, min_time)
    return json.load(open(COMPANY_DATA_FILENAME))

def load_mat_details(min_time=180):
    fetch_api_data_if_needed(f"{API_BASE_URL}/exchange/mat-details?apikey={API_KEY}", MAT_DETAILS_FILENAME, min_time)
    return json.load(open(MAT_DETAILS_FILENAME))

def load_bases_data(min_time=180):
    fetch_api_data_if_needed(f"{API_BASE_URL}/company/bases?apikey={API_KEY}", BASES_FILENAME, min_time)
    bases_data = dict((base_data["id"], base_data) for base_data in json.load(open(BASES_FILENAME)))
    HQ_LEVEL = get_hq_level(bases_data)
    return bases_data

def load_base_data(base_id):
    return json.load(open(f"data/base/{base_id}.json"))

def load_warehouse_data(warehouse_id):
    return json.load(open(f"data/warehouse/{warehouse_id}.json"))

def load_warehouses_data(min_time=180):
    fetch_api_data_if_needed(f"{API_BASE_URL}/company/warehouses?apikey={API_KEY}", WAREHOUSES_FILENAME, min_time)
    return dict((wh_data["id"], wh_data) for wh_data in json.load(open(WAREHOUSES_FILENAME)))

def load_prices_dict():
    result = {}
    for data in json.load(open(PRICES_FILENAME))["prices"]:
        result[data["matId"]] = data
    return result

def load_prices_vec():
    v = np.zeros(MAT_VECTOR_SIZE)
    for data in json.load(open(PRICES_FILENAME))["prices"]:
        cur_price = data["currentPrice"]
        if cur_price < 0:
            cur_price = 0
        v[data["matId"]] = cur_price
    return v

def load_exchange_orders(min_time=180):
    fetch_api_data_if_needed(f"{API_BASE_URL}/company/exchangeorders?apikey={API_KEY}", EXCHANGE_ORDERS_FILENAME, min_time)
    return json.load(open(EXCHANGE_ORDERS_FILENAME))

def load_exchange_data():
    return [None] + json.load(open(EXCHANGE_DATA_FILENAME))["materials"]

def get_all_base_ids(company_data):
    return [base_data["id"] for base_data in company_data["bases"]]

def get_base_warehouse_id(base_data):
    return base_data["warehouseId"]

def get_base_storage_vec(base_data, warehouses_data):
    warehouse_data = warehouses_data[base_data["warehouseId"]]
    return get_warehouse_contents_vec(warehouse_data)

def get_total_base_consumption(base_data):
    vec = np.zeros(MAT_VECTOR_SIZE)
    for mat_data in base_data["workforce"]["consumptionMaterials"]:
        vec[mat_data["matId"]] = mat_data["rate"]
    return vec

def get_warehouse_contents_vec(warehouse_data):
    vec = np.zeros(MAT_VECTOR_SIZE)
    for mat_data in warehouse_data["mats"]:
        vec[mat_data["id"]] = mat_data["am"]
    return vec

def get_company_wide_mats(warehouses_data, exchange_orders_data):
    vec = np.zeros(MAT_VECTOR_SIZE)
    for warehouse_data in warehouses_data.values():
        vec += get_warehouse_contents_vec(warehouse_data)
    for order_data in exchange_orders_data:
        vec[order_data["matId"]] += order_data["qty"]
    return vec

def get_exchange_warehouse_data(warehouses_data):
    for warehouse_data in warehouses_data.values():
        if warehouse_data["type"] == 3:
            return warehouse_data

def get_daily_consumption_vec(base_data):
    vec = np.zeros(MAT_VECTOR_SIZE)
    for consumables_data in base_data["workforce"]["consumptionMaterials"]:
        vec[consumables_data["matId"]] = consumables_data["rate"]
    return vec

def get_weight_vec():
    v = np.zeros(MAT_VECTOR_SIZE)
    for mat_id in range(1, MAT_VECTOR_SIZE):
        v[mat_id] = GAME_DATA["materials"][mat_id-1]["weight"]
    return v


def get_positive_mat_ids(mat_vec):
    return set(np.where(mat_vec - EPSILON > 0)[0].tolist())

def get_mat_id(name):
    return MAT_NAMES_TO_IDS[name]

def render_mat_with_icon(mat_name, link=False):
    asset_name = MAT_NAME_TO_ASSET_NAME.get(mat_name, "".join(mat_name.split()))
    if link == True:
        mat_text = f'<a href="https://g2.galactictycoons.com/exchange/{get_mat_id(mat_name)}">{mat_name}</a"'
    else:
        mat_text = mat_name
    return f'<span class="mat"><svg class="mat-icon"> <use href="/static/assets.svg#{asset_name}"></svg>{mat_text}</span>'

def get_time_string(days):
    if days == float("inf"):
        return "inf"
    days_int = int(days // 1)
    hours_int = int((days % 1) * 24)
    minutes_int = int((days % (1 / 24)) * 60)
    # breakpoint()
    # return f"{days_int}d {hours_int}h {minutes_int}m"
    return f"{days_int}d {hours_int}h"

def get_recipe_to_mats_matrix(include_research=True):
    m = np.zeros((RECIPE_VECTOR_SIZE, MAT_VECTOR_SIZE))
    for recipe in GAME_DATA["recipes"]:
        for input_data in recipe["inputs"]:
            m[recipe["id"], input_data["id"]] = -input_data["am"]
        output_data = recipe["output"]
        m[recipe["id"], output_data["id"]] = output_data["am"]
    return m


def get_recipe_time_vec(include_research_and_perks=True):
    company_data = load_company_data()
    bases_data = load_bases_data()
    perks_dict = get_perks_dict(company_data=company_data)

    tech_data = dict((x["id"], x["level"]) for x in company_data["technologies"])
    v = np.zeros(RECIPE_VECTOR_SIZE)
    for recipe in GAME_DATA["recipes"]:
        # In units of days
        time = recipe["timeMinutes"] / 1440
        if include_research_and_perks:
            tech_id = GAME_DATA["buildings"][recipe["producedIn"]-1]["specialization"]
            prod_multiplier = 1 + (tech_data.get(tech_id, 0) * 0.05)

            prod_multiplier += perks_dict.get(4, 0) * 0.02

            if perks_dict.get(27):
                perk_base_permits = perks_dict.get(25, 0)
                extra_base_permits = HQ_LEVEL + perk_base_permits - len(bases_data)
                prod_multiplier += extra_base_permits * 0.015

            if perks_dict.get(20, 0):
                prod_multiplier *= 1.5
            elif perks_dict.get(21, 0):
                prod_multiplier *= 0.85
            elif perks_dict.get(26, 0):
                prod_multiplier *= 1.05

            time /= (1 + ((0 + tech_data.get(tech_id, 0)) * 0.05))
        v[recipe["id"]] = time
    return v


def get_perks_dict(company_data=None):
    if not company_data:
        company_data = load_company_data()
    perks_dict = {}
    for perk_data in company_data["perks"]:
        perks_dict[perk_data["id"]] = perk_data["lvl"]
    return perks_dict

def get_hq_level(bases_data):
    for base_data in bases_data.values():
        for building_data in base_data["buildingSlots"]:
            if building_data["building"] and building_data["building"].get("type") == 9:
                return building_data["building"]["level"]

def get_recipe_building(recipe_id): 
    return GAME_DATA["buildings"][GAME_DATA["recipes"][recipe_id-1]["producedIn"]-1]["name"],


def get_building_levels_vec(base_data):
    v = np.zeros(BUILDING_VECTOR_SIZE)
    for building_data in base_data["buildingSlots"]:
        if building_data["building"] is None:
            continue
        level = building_data["building"]["level"]
        v[building_data["building"]["type"]] += level
    return v


def get_building_to_recipe_matrix():
    m = np.zeros((BUILDING_VECTOR_SIZE, RECIPE_VECTOR_SIZE))
    for recipe in GAME_DATA["recipes"]:
        m[recipe["producedIn"]][recipe["id"]] = 1
    return m


def get_material_name(material_id):
    return GAME_DATA["materials"][material_id-1]["name"]


def get_building_name(building_id):
    return GAME_DATA["buildings"][building_id-1]["name"]


def get_recipe_output_name(recipe_id):
    output_mat_id = GAME_DATA["recipes"][recipe_id-1]["output"]["id"]
    return get_material_name(output_mat_id)


def get_production_buildings(building_to_recipe_matrix):
    return np.where(building_to_recipe_matrix.sum(1))[0]


def format_price(price):
    return f"{price:.2f}".rstrip('0').rstrip('.')

# def get_base_names_to_ids():
#     return dict((base_data["name"], base_data["id"]) for base_data in COMPANY_DATA["bases"])

GAME_DATA = json.load(open("data/game_data.json"))
COMPANY_DATA = load_company_data()

MAT_NAMES_TO_IDS = {}
for mat_data in GAME_DATA["materials"]:
    MAT_NAMES_TO_IDS[mat_data["name"]] = mat_data["id"]

MAT_VECTOR_SIZE = len(GAME_DATA["materials"]) + 1
RECIPE_VECTOR_SIZE = len(GAME_DATA["recipes"]) + 1
BUILDING_VECTOR_SIZE = len(GAME_DATA["buildings"]) + 1

EPSILON = 1e-4