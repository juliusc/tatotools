import json
import os
import requests
import time

from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd


GAME_DATA = json.load(open("data/game_data.json"))
RECIPE_ID_TO_RECIPES = dict((recipe["id"], recipe) for recipe in GAME_DATA["recipes"])

MAT_NAMES_TO_IDS = {}
for mat_data in GAME_DATA["materials"]:
    MAT_NAMES_TO_IDS[mat_data["name"]] = mat_data["id"]

MAT_VECTOR_SIZE = len(GAME_DATA["materials"]) + 1
RECIPE_VECTOR_SIZE = len(GAME_DATA["recipes"]) + 1
BUILDING_VECTOR_SIZE = len(GAME_DATA["buildings"]) + 1

EPSILON = 1e-4

MAT_NAME_TO_ASSET_NAME = {
    "Rations": "BasicRations",
    "Tools": "BasicTools",
    "Exosuit": "BasicExosuit",
    "Bio-Nutrient Blend": "NutrientBlend",
}
RECIPE_ID_TO_CUSTOM_NAMES = {
    5: "Iron Ore (Oxygen)",
    196: "Iron Ore (Flux)",
    11: "Rations (Fruits)",
    33: "Rations (Meat)",
    21: "Fruits (No Fertilizer)",
    27: "Fruits (Fertilizer)",
    22: "Vegetables (No Fertilizer)",
    28: "Vegetables (Fertilizer)",
    23: "Construction Kit (Construction Vehicle)",
    133: "Construction Kit (No Construction Vehicle)",
    30: "Grain (No Fertilizer)",
    60: "Grain (Fertilizer)",
    42: "Titanium Ore (Drill)",
    148: "Titanium Ore (TiC Drill)",
    47: "Wood (No Fertilizer)",
    140: "Wood (Fertilizer)",
    51: "Fabric (Cotton)",
    52: "Fabric (Polyethylene)",
    53: "Coffee Beans (No Fertilizer)",
    142: "Coffee Beans (Fertilizer)",
    68: "Research Data (Construction)",
    70: "Research Data (Agriculture)",
    71: "Research Data (Extraction)",
    94: "Research Data (Manufacturing)",
    74: "Pipes (Iron)",
    75: "Pipes (Polyethylene)",
    81: "Advanced Research Data (Copper)",
    134: "Advanced Research Data (Kevlar)",
    83: "Platinum Ore (Drill)",
    149: "Platinum Ore (TiC Drill)",
    105: "Apex Research Data (Bioxene)",
    135: "Apex Research Data (Sensor Array)",
    136: "Honeycaps (No Fertilizer)",
    141: "Honeycaps (Fertilizer)",
    150: "Ship Repair Kit (Aluminium)",
    183: "Ship Repair Kit (Hydrogen Generator)",
    195: "Ship Repair Kit (Fission Reactor)",
    45: "Quantum Research Data (Antimatter)",
    180: "Quantum Research Data (Quadranium)",
    189: "Quantum Research Data (Biopolyne)",
}

EPSILON = 1e-3

API_BASE_URL = "https://api.g2.galactictycoons.com/public"


DEBUG_MODE = True
DEBUG_MODE = False
CACHE_FILE = "cache.json"
if Path(CACHE_FILE).exists():
    CACHE = json.load(open(CACHE_FILE))
else:
    CACHE = {}

class NotATSError(Exception):
    status_code = 460
    msg = "You must be a member of ATS."


class InvalidApiKeyError(Exception):
    status_code = 461
    msg = "Invalid API key."


class EmptyApiKeyError(Exception):
    status_code = 462
    msg = "Go to API keys to set your extended API key."


class RateLimitExceededError(Exception):
    status_code = 463
    msg = "Rate limit exceeded"


def handle_gt_api_response(response):
    if response.status_code == 429:
        raise RateLimitExceededError("Rate limit exceeded")
    elif response.status_code == 401:
        raise InvalidApiKeyError("Invalid or empty API key. Go 'API key' to set your extended API key.")
    elif response.status_code != 200:
        raise Exception("Unhandled GT API error")


def fetch_url(url):
    if DEBUG_MODE and url in CACHE:
        return CACHE[url]
    
    response = requests.get(url)
    handle_gt_api_response(response)
    response_json = response.json()
    if DEBUG_MODE:
        CACHE[url] = response_json
        json.dump(CACHE, open(CACHE_FILE, "w"))
    return response_json


def fetch_api_data_if_needed(url, save_location, min_time=300):
    if not os.path.exists(save_location) or time.time() - os.path.getmtime(save_location) > min_time:
        fetch_url(url, save_location)


def load_company_and_bases_data(api_key):
    company_data = load_company_data(api_key)
    bases_data = load_bases_data(api_key)
    company_data["hqLevel"] = get_hq_level(bases_data)
    return company_data, bases_data


def load_company_data(api_key):
    company_data = fetch_url(f"{API_BASE_URL}/company?apikey={api_key}")
    if company_data["gId"] != 39:
        raise NotATSError("You must be a member of ATS")
    return company_data


def load_mat_details(min_time=180):
    fetch_api_data_if_needed(f"{API_BASE_URL}/exchange/mat-details?apikey={API_KEY}", MAT_DETAILS_FILENAME, min_time)
    return json.load(open(MAT_DETAILS_FILENAME))

def load_bases_data(api_key):
    data = fetch_url(f"{API_BASE_URL}/company/bases?apikey={api_key}")
    bases_data = dict((base_data["id"], base_data) for base_data in data)
    return bases_data


def load_warehouses_data(api_key):
    data = fetch_url(f"{API_BASE_URL}/company/warehouses?apikey={api_key}")
    return dict((wh_data["id"], wh_data) for wh_data in data)


def load_prices_dict(min_time=180):
    fetch_api_data_if_needed(f"{API_BASE_URL}/exchange/mat-prices?apikey={API_KEY}", PRICES_FILENAME, min_time)
    result = {}
    for data in json.load(open(PRICES_FILENAME))["prices"]:
        result[data["matId"]] = data
    return result

def load_prices_vec(min_time=180):
    fetch_api_data_if_needed(f"{API_BASE_URL}/exchange/mat-prices?apikey={API_KEY}", PRICES_FILENAME, min_time)
    v = np.zeros(MAT_VECTOR_SIZE)
    for data in json.load(open(PRICES_FILENAME))["prices"]:
        cur_price = data["currentPrice"]
        if cur_price < 0:
            cur_price = 0
        v[data["matId"]] = cur_price
    return v


def load_exchange_orders(api_key):
    return fetch_url(f"{API_BASE_URL}/company/exchangeorders?apikey={api_key}")


def load_exchange_data(api_key):
    raw_data = fetch_url(f"{API_BASE_URL}/exchange/mat-details?apikey={api_key}")
    return [None] + raw_data["materials"]

def get_all_base_ids(company_data):
    return [base_data["id"] for base_data in company_data["bases"]]

def get_base_warehouse_id(base_data):
    return base_data["warehouseId"]

def get_base_storage_vec(base_data, warehouses_data):
    warehouse_data = warehouses_data[base_data["warehouseId"]]
    return get_warehouse_contents_vec(warehouse_data)

def get_base_storage_matrix(base_ids, bases_data, warehouses_data):
    """Return a matrix of shape (base ID, material ID)"""
    m = np.zeros((len(base_ids), MAT_VECTOR_SIZE), dtype=int)
    for i, base_id in enumerate(base_ids):
        warehouse_data = warehouses_data[bases_data[base_id]["warehouseId"]]
        for mat_data in warehouse_data["mats"]:
            m[i, mat_data["id"]] = mat_data["am"]
    return m


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

def get_exchange_mats(exchange_orders_data):
    vec = np.zeros(MAT_VECTOR_SIZE)
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

def get_daily_consumption_matrix(base_ids, bases_data):
    m = np.zeros((len(base_ids), MAT_VECTOR_SIZE))
    for i, base_id in enumerate(base_ids):
        for consumables_data in bases_data[base_id]["workforce"]["consumptionMaterials"]:
            m[i][consumables_data["matId"]] = consumables_data["rate"]
    return m

def get_weight_vec():
    v = np.zeros(MAT_VECTOR_SIZE)
    for mat_id in range(1, MAT_VECTOR_SIZE):
        v[mat_id] = GAME_DATA["materials"][mat_id-1]["weight"]
    return v


def get_positive_mat_ids(mat_vec):
    return set(np.where(mat_vec - EPSILON > 0)[0].tolist())

def get_mat_id(name):
    return MAT_NAMES_TO_IDS[name]

# def render_mat_with_icon(mat_name, link=False):
#     asset_name = MAT_NAME_TO_ASSET_NAME.get(mat_name, "".join(mat_name.split()))
#     if link == True:
#         mat_text = f'<a href="https://g2.galactictycoons.com/exchange/{get_mat_id(mat_name)}">{mat_name}</a"'
#     else:
#         mat_text = mat_name
#     return f'<span class="mat"><svg class="mat-icon"> <use href="/static/assets.svg#{asset_name}"></svg>{mat_text}</span>'

def render_mat_with_icon(mat_id, link=False, name_override=None):
    if name_override:
        mat_name = name_override
    else:
        mat_name = get_material_name(mat_id)
    asset_name = MAT_NAME_TO_ASSET_NAME.get(mat_name, "".join(mat_name.split()))
    if link == True:
        mat_text = f'<a href="https://g2.galactictycoons.com/exchange/{get_mat_id(mat_name)}">{mat_name}</a"'
    else:
        mat_text = mat_name
    return f'<span class="mat"><svg class="mat-icon"> <use href="/static/assets.svg#{asset_name}"></svg>{mat_text}</span>'


def render_recipe_with_icon(recipe_id, link=False):
    mat_id = get_recipe_output_id(recipe_id)
    return render_mat_with_icon(mat_id, link=link, name_override=get_recipe_name(recipe_id))


def render_weight(weight):
    return f"{weight:.2f}".rstrip("0").rstrip(".")


def get_time_string(days):
    if days == float("inf"):
        return "inf"
    total_minutes = int(round(days * 24 * 60))
    days_int, rem = divmod(total_minutes, 24 * 60)
    hours_int, minutes_int = divmod(rem, 60)
    return f"{days_int}d {hours_int}h {minutes_int}m"


def get_recipe_to_mats_matrix(include_research=True):
    m = np.zeros((RECIPE_VECTOR_SIZE, MAT_VECTOR_SIZE), dtype=int)
    for recipe in GAME_DATA["recipes"]:
        for input_data in recipe["inputs"]:
            m[recipe["id"], input_data["id"]] = -input_data["am"]
        output_data = recipe["output"]
        m[recipe["id"], output_data["id"]] = output_data["am"]
    return m


def get_recipe_time_vec(company_data, bases_data, include_research_and_perks=True):
    perks_dict = get_perks_dict(company_data=company_data)

    tech_data = dict((x["id"], x["level"]) for x in company_data["technologies"])
    v = np.zeros(RECIPE_VECTOR_SIZE)
    for recipe in GAME_DATA["recipes"]:
        # print(GAME_DATA["materials"][recipe["output"]["id"]]["name"])
        # In units of days
        time = recipe["timeMinutes"] / 1440
        if include_research_and_perks:
            tech_id = GAME_DATA["buildings"][recipe["producedIn"]-1]["specialization"]
            tech_bonus = tech_data.get(tech_id, 0) * 0.05
            # print("tech", tech_bonus)

            workflow_optimization_bonus = perks_dict.get(4, 0) * 0.02
            # print("workflow optimization", workflow_optimization_bonus)

            consolidated_operations_bonus = 0
            if perks_dict.get(27):
                bobs_permits = perks_dict.get(25, 0)
                extra_base_permits = company_data["hqLevel"] + bobs_permits - get_base_permits_used(bases_data)
                consolidated_operations_bonus += extra_base_permits * 0.015
            # print("consolidated operations", consolidated_operations_bonus)

            supervision_bonus = 0
            if perks_dict.get(20, 0):
                supervision_bonus = 1.25
            elif perks_dict.get(21, 0):
                supervision_bonus = 0.85
            elif perks_dict.get(26, 0):
                supervision_bonus = 1.05
            # print("supervision bonus", supervision_bonus)

            prod_multiplier = (1 + tech_bonus + workflow_optimization_bonus + consolidated_operations_bonus) * supervision_bonus
            # print("final bonus", prod_multiplier)

            time /= prod_multiplier
        v[recipe["id"]] = time
    v[0] = 100000000
    return v


def get_base_permits_used(bases_data):
    used = 0
    for base_data in bases_data.values():
        if len(base_data["buildingSlots"]) == 25:
            used += 1
        elif len(base_data["buildingSlots"]) == 40:
            used += 2
        elif len(base_data["buildingSlots"]) == 50:
            used += 3
    return used


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

def get_recipe_name(recipe_id):
    if recipe_id in RECIPE_ID_TO_CUSTOM_NAMES:
        return RECIPE_ID_TO_CUSTOM_NAMES[recipe_id]
    else:
        return get_material_name(RECIPE_ID_TO_RECIPES[recipe_id]["output"]["id"])

def get_is_prod_building_vec():
    v = np.zeros(BUILDING_VECTOR_SIZE)
    for recipe_data in GAME_DATA["buildings"]:
        if recipe_data.get("recipesIds"):
            v[recipe_data["id"]] = 1
    return v


def get_building_levels_vec(base_data, prod_buildings_only=False):
    is_prod_building_vec = get_is_prod_building_vec()
    v = np.zeros(BUILDING_VECTOR_SIZE)
    for building_data in base_data["buildingSlots"]:
        if building_data["building"] is None:
            continue
        level = building_data["building"]["level"]
        v[building_data["building"]["type"]] += level
        if prod_buildings_only:
            v *= is_prod_building_vec
    return v


def get_building_levels_matrix(base_ids, bases_data, prod_buildings_only=False):
    m = np.zeros((len(base_ids), BUILDING_VECTOR_SIZE))
    for i, base_id in enumerate(base_ids):
        m[i] = get_building_levels_vec(bases_data[base_id], prod_buildings_only=prod_buildings_only)
    return m


def get_building_to_recipe_matrix():
    m = np.zeros((BUILDING_VECTOR_SIZE, RECIPE_VECTOR_SIZE))
    for recipe in GAME_DATA["recipes"]:
        m[recipe["producedIn"]][recipe["id"]] = 1
    m[0, 0] = 1
    return m


def get_material_name(material_id):
    return GAME_DATA["materials"][material_id-1]["name"]


def get_building_name(building_id):
    return GAME_DATA["buildings"][building_id-1]["name"]


def get_recipe_output_id(recipe_id):
    return GAME_DATA["recipes"][recipe_id-1]["output"]["id"]


def get_recipe_output_name(recipe_id):
    return get_material_name(get_recipe_output_id(recipe_id))


def get_production_buildings(building_to_recipe_matrix):
    return np.where(building_to_recipe_matrix.sum(1))[0]


def format_price(price):
    return f"{price:.2f}".rstrip('0').rstrip('.')


def iterate_exchange_orders(exchange_data):
    for mat_id, mat_data in enumerate(exchange_data):
        if mat_id == 0:
            continue
        yield mat_id, mat_data


def get_exchange_orders_matrices(exchange_data):
    max_sell_orders = 0
    for _, mat_data in iterate_exchange_orders(exchange_data):
        max_sell_orders = max(max_sell_orders, len(mat_data["orders"]))
    qty_matrix = np.zeros((MAT_VECTOR_SIZE, max_sell_orders))
    price_matrix = np.zeros((MAT_VECTOR_SIZE, max_sell_orders))
    for mat_id, mat_data in iterate_exchange_orders(exchange_data):
        for i, order in enumerate(mat_data["orders"]):
            price_matrix[mat_id, i] = order["unitPrice"]
            qty_matrix[mat_id, i] = order["qty"]
    return price_matrix, qty_matrix


def get_avg_qty_sold_vec(exchange_data):
    vec = np.zeros(MAT_VECTOR_SIZE)
    for mat_id, mat_data in iterate_exchange_orders(exchange_data):
        vec[mat_id] = mat_data["avgQtySoldDaily"]
    return vec


def get_current_prices_vec(exchange_data):
    vec = np.zeros(MAT_VECTOR_SIZE)
    for mat_id, mat_data in iterate_exchange_orders(exchange_data):
        if mat_data["matName"] == "TEMP":
            vec[mat_id] = 0.000001
        else:
            vec[mat_id] = mat_data["currentPrice"]
    vec[0] = 100000
    return vec


def prepare_wishlist_additems_body(data):
    # Data is a dict of (base_id, dict of (mat_id, amount))
    result = []
    for base_id, base_data in data.items():
        mats = []
        for mat_id, amt in base_data.items():
            mats.append({"id": mat_id, "am": amt})
        result.append({
            "wishlistId": base_id,
            "mats": mats
        })
    return result
