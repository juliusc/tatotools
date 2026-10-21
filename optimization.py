from collections import defaultdict

import cvxpy as cp
import numpy as np
import pandas as pd

import lib

EPSILON = 1e-3


def prepare_report(base_ids, plan_orders, consumables_duration, company_data, bases_data, warehouses_data):
    recipe_to_mats_matrix = lib.get_recipe_to_mats_matrix()
    building_to_recipe_matrix = lib.get_building_to_recipe_matrix()
    recipe_time_vec = lib.get_recipe_time_vec(company_data, bases_data)
    weights_vec = lib.get_weight_vec()

    bases_warehouse_mat = lib.get_base_storage_matrix(base_ids, bases_data, warehouses_data)
    buildings_mat = lib.get_building_levels_matrix(base_ids, bases_data, prod_buildings_only=True)
    exchange_warehouse_vec = lib.get_warehouse_contents_vec(lib.get_exchange_warehouse_data(warehouses_data))
    consumption_mat = lib.get_daily_consumption_matrix(base_ids, bases_data) * consumables_duration

    base_idxs_to_names = dict((idx, bases_data[base_ids[idx]]["name"]) for idx in range(len(base_ids)))

    summary_purchases = defaultdict(int)
    summary_purchase_costs = defaultdict(int)
    summary_orders = defaultdict(int)
    wishlist_data = defaultdict(dict)
    result = {
        "bases": {},
        "summary": {"purchases": []},
        "wishlist": {}
    }
    for base_id in base_ids:
        result["bases"][base_id] = {
            "name": bases_data[base_id]["name"],
            "orders": [], "materials": [], "prod_times": []
        }

    for base_id_idx, recipe_id in zip(*np.where(plan_orders > 0)):
        result["bases"][base_ids[base_id_idx]]["orders"].append({
            "building": lib.GAME_DATA["buildings"][lib.GAME_DATA["recipes"][recipe_id-1]["producedIn"]-1]["name"],
            "recipe": lib.render_recipe_with_icon(recipe_id),
            "amount": int(plan_orders[base_id_idx, recipe_id])
        })
        summary_orders[lib.render_recipe_with_icon(recipe_id)] += int(plan_orders[base_id_idx, recipe_id])

    inputs_needed_mat = -np.minimum(
        np.matmul(plan_orders, recipe_to_mats_matrix) + bases_warehouse_mat - consumption_mat, 0)

    for base_id_idx, mat_id in zip(*np.where(inputs_needed_mat > 0)):
        mat_id = int(mat_id)
        import_amount = int(inputs_needed_mat[base_id_idx, mat_id])
        result["bases"][base_ids[base_id_idx]]["materials"].append({
            # "material": lib.render_mat_with_icon(lib.get_material_name(mat_id), link=False), 
            "material": lib.render_mat_with_icon(mat_id),
            "at_base": int(bases_warehouse_mat[base_id_idx, mat_id]),
            "import_amount": import_amount,
            "import_weight": lib.render_weight(int(inputs_needed_mat[base_id_idx, mat_id]) * weights_vec[mat_id])
        })
        wishlist_data[bases_data[base_ids[base_id_idx]]["planetId"]][mat_id] = import_amount

    orders_time = cp.multiply(plan_orders, recipe_time_vec)
    buildings_time = cp.matmul(orders_time, building_to_recipe_matrix.T)

    actuals_building_time = buildings_time.value / buildings_mat
    for base_id_idx, building_id in zip(*np.where(buildings_mat > 0)):
        result["bases"][base_ids[base_id_idx]]["prod_times"].append({
            "building": lib.get_building_name(building_id),
            "duration": lib.get_time_string(actuals_building_time[base_id_idx, building_id])
        })

    purchases_vec = inputs_needed_mat.sum(0) - exchange_warehouse_vec
    for mat_id in np.where(purchases_vec > 0)[0]:
        result["summary"]["purchases"].append({
            "material": lib.get_material_name(mat_id),
            "amount": purchases_vec[mat_id]
        })
    result["wishlist"] = lib.prepare_wishlist_additems_body(wishlist_data)
    return result


def plan_production2(base_ids,
                    company_data,
                    bases_data,
                    warehouses_data,
                    budget,
                    max_prod_duration,
                    exchange_data,
                    exchange_orders_data,
                    consumables_duration=None,
                    max_weight=None,
                    include_consumables=True,
                    include_exchange=True,
                    no_inputs_sellable=True,
                    only_outputs_sellable=True,
                    no_outputs_purchaseable=True,
                    max_sellable_volume_pct=0.25,
                    roi=0.02,
                    num_price_segments=12,
                    max_rungs=24.0):
    recipe_to_mats_matrix = lib.get_recipe_to_mats_matrix()
    recipe_time_vec = lib.get_recipe_time_vec(company_data, bases_data)
    building_to_recipe_matrix = lib.get_building_to_recipe_matrix()
    weights_vec = lib.get_weight_vec()

    # Company-specific data
    bases_warehouse_mat = lib.get_base_storage_matrix(base_ids, bases_data, warehouses_data)
    exchange_warehouse_vec = lib.get_warehouse_contents_vec(lib.get_exchange_warehouse_data(warehouses_data))
    exchange_warehouse_vec += lib.get_exchange_mats(exchange_orders_data)
    buildings_mat = lib.get_building_levels_matrix(base_ids, bases_data, prod_buildings_only=True)
    daily_consumption_mat = lib.get_daily_consumption_matrix(base_ids, bases_data)

    # Exchange data
    exchange_price_mat, exchange_qtys_mat = lib.get_exchange_orders_matrices(exchange_data)
    sales_limit = lib.get_avg_qty_sold_vec(exchange_data)
    sales_prices = exchange_price_mat[:, 0] * 0.99

    # Variables
    orders_var = cp.Variable((len(base_ids), recipe_to_mats_matrix.shape[0]))
    purchases_var = cp.Variable(exchange_qtys_mat.shape)
    sales_var = cp.Variable(lib.MAT_VECTOR_SIZE)
    exchange_sales_var = cp.Variable(lib.MAT_VECTOR_SIZE)
    total_sales = sales_var + exchange_sales_var
    shipping_var = cp.Variable(bases_warehouse_mat.shape)

    # --- Piecewise-linear price-ladder sales revenue ---
    # True ladder: R(s) = p0 * k * (1 - r^(-s/k)) / ln r, k = rung capacity.
    # R is concave => lies above its chords: model revenue with secant
    # segments via an epigraph -> pure LP, solvable with HiGHS.
    log_roi = np.log(1 + roi)
    rung_capacity = (sales_limit + 1) * max_sellable_volume_pct   # (n_mats,)

    fracs = np.linspace(0.0, max_rungs, num_price_segments + 1)   # fraction of a rung
    s_brk = np.outer(rung_capacity, fracs)                        # (n_mats, B+1) qty breakpoints
    v_brk = (sales_prices * rung_capacity / log_roi)[:, None] \
        * (1.0 - np.exp(-log_roi * fracs))[None, :]               # (n_mats, B+1) revenue at breakpoints

    revenue_var = cp.Variable(lib.MAT_VECTOR_SIZE)
    sales_revenue = cp.sum(revenue_var)

    if consumables_duration is None:
        consumables_duration = max_prod_duration
    resulting_mats = (
        bases_warehouse_mat
        + cp.matmul(orders_var, recipe_to_mats_matrix)
        + shipping_var
        - daily_consumption_mat * max_prod_duration
    )

    orders_time = cp.multiply(orders_var, recipe_time_vec)
    buildings_time = cp.matmul(orders_time, building_to_recipe_matrix.T)
    max_building_time = buildings_mat * consumables_duration

    total_purchase_cost = cp.sum(cp.multiply(purchases_var, exchange_price_mat))

    objective = cp.Maximize(sales_revenue - cp.sum(total_purchase_cost) * (1 + roi) ** max_prod_duration)

    constraints = [
        orders_var >= 0,
        buildings_time <= max_building_time,
        resulting_mats >= 0,
        purchases_var >= 0,
        purchases_var <= exchange_qtys_mat,
        sales_var >= 0,
        sales_var <= resulting_mats.sum(0),
        exchange_sales_var >= 0,
        # exchange_sales_var <= exchange_warehouse_vec - shipping_var.sum(0) + purchases_var.sum(1),
        # Stay within the modeled ladder range (past the last chord a
        # piecewise model of a concave function would overestimate)
        total_sales <= s_brk[:, -1],
        shipping_var >= 0,
        # shipping_var.sum(0) <= exchange_warehouse_vec + purchases_var.sum(1),
        exchange_sales_var + shipping_var.sum(0) <= exchange_warehouse_vec + purchases_var.sum(1)
    ]

    # Revenue epigraph: revenue_var <= each chord of R(sales_var).
    # Maximizing pushes revenue_var onto the lower envelope of the chords,
    # which is the piecewise-linear (underestimating) approximation of R.
    for b in range(num_price_segments):
        slope = (v_brk[:, b + 1] - v_brk[:, b]) / (s_brk[:, b + 1] - s_brk[:, b])
        intercept = v_brk[:, b] - slope * s_brk[:, b]
        constraints.append(revenue_var <= cp.multiply(slope, total_sales) + intercept)

    if budget is not None and budget >= 0:
        constraints.append(total_purchase_cost <= budget * 100)

    if max_weight is not None:
        constraints.append(cp.sum(cp.multiply(shipping_var, weights_vec), axis=1) <= max_weight)

    prob = cp.Problem(objective, constraints)
    solution = prob.solve(solver=cp.HIGHS, verbose=True)
    # breakpoint()

    result = {"bases": {}, "summary": {"purchases": []}}
    for base_id in base_ids:
        result["bases"][base_id] = {"orders": [], "materials": [], "prod_times": []}

    # Heuristic rounding
    factor = (orders_var.value.max() / 1000)
    final_orders = (np.floor(orders_var.value / factor) * factor).astype(int)
    return final_orders


def get_full_production_time(base_id, bases_data, company_data, warehouses_data, limit_by_prod_orders=False):
    base_data = bases_data[base_id]
    recipe_to_mats_matrix = lib.get_recipe_to_mats_matrix()
    recipe_time_vec = lib.get_recipe_time_vec(company_data, bases_data)
    building_to_recipe_matrix = lib.get_building_to_recipe_matrix()
    storage_vec = lib.get_base_storage_vec(base_data, warehouses_data)
    buildings_vec = lib.get_building_levels_vec(base_data)
   
    owned_buildings_idxs = np.intersect1d(np.where(buildings_vec)[0], lib.get_production_buildings(building_to_recipe_matrix))
    buildings_vec_subset = buildings_vec[owned_buildings_idxs]

    possible_recipe_idxs = np.where(np.matmul(buildings_vec_subset, building_to_recipe_matrix[owned_buildings_idxs]))[0]

    buildings_to_recipe_matrix_subset = building_to_recipe_matrix[owned_buildings_idxs][:, possible_recipe_idxs]
    recipe_to_mats_matrix_subset = recipe_to_mats_matrix[possible_recipe_idxs]
    recipe_time_vec_subset = recipe_time_vec[possible_recipe_idxs]

    orders_var = cp.Variable(possible_recipe_idxs.shape[0], integer=True)
    resulting_mats = (cp.matmul(orders_var, recipe_to_mats_matrix_subset)) + storage_vec

    orders_time = cp.multiply(orders_var, recipe_time_vec_subset)
    orders_time_per_building_type = cp.matmul(orders_time, buildings_to_recipe_matrix_subset.T)

    objective = cp.Maximize((orders_time_per_building_type / buildings_vec_subset).min() + orders_var.sum() * EPSILON)

    constraints = [
        orders_var >= 0,
        resulting_mats >= 0,
        orders_time_per_building_type / buildings_vec_subset <= 10000
    ]
    if limit_by_prod_orders:
        order_limit = np.zeros(lib.RECIPE_VECTOR_SIZE)
        for prod_order in base_data["productionOrders"]:
            if prod_order["amt"] == 65535:
                order_limit[prod_order["rId"]] = float("inf")
            else:
                order_limit[prod_order["rId"]] += prod_order["amt"]
        constraints.append(orders_var <= order_limit[possible_recipe_idxs])

    prob = cp.Problem(objective, constraints)

    result = prob.solve()

    order_hours_df = pd.DataFrame(columns=["Building", "Hours"])
    for i in range(buildings_vec_subset.shape[0]):
        orders_time = orders_time_per_building_type.value[i]
        order_hours_df.loc[len(order_hours_df)] = [
            lib.get_building_name(owned_buildings_idxs[i]),
            orders_time / buildings_vec_subset[i] * 24
        ]

    orders_df = pd.DataFrame(columns=["Building", "Recipe", "Num orders"])

    for i in range(orders_var.value.size):
        recipe_id = possible_recipe_idxs[i]
        if orders_var.value[i] > 0:
            orders_df.loc[len(orders_df)] = [
                lib.GAME_DATA["buildings"][lib.GAME_DATA["recipes"][recipe_id-1]["producedIn"]-1]["name"],
                lib.get_recipe_output_name(recipe_id),
                int(orders_var.value[i])
            ]
    return (orders_time_per_building_type.value / buildings_vec_subset).min()

    # return result