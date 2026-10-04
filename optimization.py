import cvxpy as cp
import numpy as np
import pandas as pd

import lib

EPSILON = 1e-3


def plan_production2(selected_bases,
                     durations,
                     budget,
                     period,
                     bases_data,
                     warehouses_data,
                     exchange_data,
                     include_consumables=True,
                     include_exchange=True,
                     no_inputs_sellable=True,
                     only_outputs_sellable=True,
                     no_outputs_purchaseable=True):
    recipe_to_mats_matrix = lib.get_recipe_to_mats_matrix()
    recipe_time_vec = lib.get_recipe_time_vec()
    building_to_recipe_matrix = lib.get_building_to_recipe_matrix()
    base_warehouse_vec = lib.get_base_storage_vec(base_data, warehouses_data)
    exchange_warehouse_vec = lib.get_warehouse_contents_vec(lib.get_exchange_warehouse_data(warehouses_data))

    exchange_orders_data = lib.load_exchange_orders()
    other_warehouses_mat_vec = lib.get_company_wide_mats(warehouses_data, exchange_orders_data)
    other_warehouses_mat_vec -= base_warehouse_vec

    # storage_vec[lib.get_mat_id("Wood")] = 20000
    # storage_vec[lib.get_mat_id("Carbon")] = 0
    buildings_vec = lib.get_building_levels_vec(base_data)
    weights_vec = lib.get_weight_vec()
    daily_consumption_vec = lib.get_daily_consumption_vec(base_data)

    owned_buildings_idxs = np.intersect1d(np.where(buildings_vec)[0], lib.get_production_buildings(building_to_recipe_matrix))
    buildings_vec_subset = buildings_vec[owned_buildings_idxs]

    possible_recipe_idxs = np.where(np.matmul(buildings_vec_subset, building_to_recipe_matrix[owned_buildings_idxs]))[0]

    buildings_to_recipe_matrix_subset = building_to_recipe_matrix[owned_buildings_idxs][:, possible_recipe_idxs]
    recipe_to_mats_matrix_subset = recipe_to_mats_matrix[possible_recipe_idxs]
    recipe_time_vec_subset = recipe_time_vec[possible_recipe_idxs]

    possible_mats_ids = np.where(np.logical_or((recipe_to_mats_matrix_subset != 0).any(0), daily_consumption_vec != 0))[0]
    recipe_to_mats_matrix_subset = recipe_to_mats_matrix_subset[:, possible_mats_ids]
    base_warehouse_vec_subset = base_warehouse_vec[possible_mats_ids]
    exchange_warehouse_vec_subset = exchange_warehouse_vec[possible_mats_ids]
    daily_consumption_vec_subset = daily_consumption_vec[possible_mats_ids]
    weights_vec_subset = weights_vec[possible_mats_ids]
    other_warehouses_mat_vec_subset = other_warehouses_mat_vec[possible_mats_ids]

    max_sell_orders = 0
    for mat_id in possible_mats_ids:
        max_sell_orders = max(max_sell_orders, len(exchange_data[mat_id]["orders"]))

    # Variables

    orders_var = cp.Variable(possible_recipe_idxs.shape[0], integer=True)
    purchases_var = cp.Variable([len(possible_mats_ids), max_sell_orders], integer=True)
    
    purchase_costs = np.zeros(purchases_var.shape)
    purchase_qtys = np.zeros(purchases_var.shape)
    for i, mat_id in enumerate(possible_mats_ids):
        for j, sell_order in enumerate(exchange_data[mat_id]["orders"]):
            purchase_costs[i, j] = sell_order["unitPrice"]
            purchase_qtys[i, j] = sell_order["qty"]
    prices_vec_subset = purchase_costs[:, 0]

    NUM_BUCKETS = 10
    sales_var = cp.Variable([len(possible_mats_ids), NUM_BUCKETS], integer=True)
    sales_prices = np.zeros(sales_var.shape)
    sales_max_amts = np.zeros(sales_var.shape)
    for i, mat_id in enumerate(possible_mats_ids):
        sales_max_amts[i] = exchange_data[mat_id]["avgQtySoldDaily"] / NUM_BUCKETS * 2 #/ 2 # Divide by 2 is arbitrary
        for j in range(sales_prices.shape[1]):
            sales_prices[i, j] = exchange_data[mat_id]["currentPrice"] * (0.8 + ((j + 1) * 0.025))
    sales_revenue = cp.sum(cp.multiply(sales_var, sales_prices))

    input_mat_mask = (recipe_to_mats_matrix_subset < 0).any(0)
    output_mat_mask = (recipe_to_mats_matrix_subset > 0).any(0)
    if no_inputs_sellable:
        sales_prices *= np.expand_dims(~(np.logical_and(input_mat_mask, ~output_mat_mask)), axis=1)
    # if no_outputs_purchaseable:
    #     purchase_qtys *= np.expand_dims(~output_mat_mask, axis=1)

    def get_possible_idx(name):
        return np.where(possible_mats_ids == lib.get_mat_id(name))[0][0]

    # sales_prices[get_possible_idx("Ethanol")] *= 100
    # prices_vec_subset[get_possible_idx("Biopolyne")] *= 3
    # purchase_qtys[get_possible_idx("Biopolyne")] = 0


    total_storage_vec_subset = base_warehouse_vec_subset
    # if include_exchange:
    #     total_storage_vec_subset = total_storage_vec_subset + exchange_warehouse_vec_subset

    resulting_mats = (cp.matmul(orders_var, recipe_to_mats_matrix_subset)) + total_storage_vec_subset + purchases_var.sum(1)
    if consumables_duration is None:
        consumables_duration = period
    if include_consumables:
        needs_consumbles = (total_storage_vec_subset - daily_consumption_vec_subset * period < 0).any()
        if needs_consumbles:
            resulting_mats = resulting_mats - daily_consumption_vec_subset * consumables_duration

    orders_time = cp.multiply(orders_var, recipe_time_vec_subset)
    orders_time_per_building_type = cp.matmul(orders_time, buildings_to_recipe_matrix_subset.T)

    total_purchase_cost = cp.sum(cp.multiply(purchases_var, purchase_costs))

    total_value = cp.sum(cp.multiply(resulting_mats, prices_vec_subset))

    # objective = cp.Maximize(cp.sum(total_value) - cp.sum(total_purchase_cost) * 1.1)
    objective = cp.Maximize(sales_revenue - cp.sum(total_purchase_cost) * 1.1)

    constraints = [
        # Orders are non-negative
        orders_var >= 0,
        # Goods are non-negative after production
        resulting_mats >= 0,
        # Goods after sales is non-negative
        resulting_mats + other_warehouses_mat_vec_subset - cp.sum(sales_var, 1) >= 0,
        # Purchases are non-negative
        purchases_var >= 0,
        # Purchases come from existing sell orders
        purchases_var <= purchase_qtys,
        # Orders match available building production levels
        orders_time_per_building_type / buildings_vec_subset <= period,
        # Sales limit
        sales_var <= sales_max_amts,
        # Sales are non-negative
        sales_var >= 0,
        # # No purchased goods are sold
        # cp.sum(cp.multiply(purchases_var.sum(1), sales_var.sum(1))) == 0

        # Only researched recipes used
        # cp.sum(cp.multiply(orders_var, (1 - recipe_is_doable.astype(float)))) == 0,
        # cp.sum(cp.multiply(lib.get_weight_vec(), purchases_var)) <= 6000
    ]


    # Purchases are within budget
    if budget is not None and budget >= 0:
        constraints.append(total_purchase_cost <= budget * 100)
    # breakpoint()

    if max_weight is not None:
        constraints.append(cp.sum(cp.multiply(purchases_var.sum(1), weights_vec_subset)) <= max_weight)

    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    # result = prob.solve()

    order_hours_df = pd.DataFrame(columns=["Building", "Hours"])
    for i in range(buildings_vec_subset.shape[0]):
        orders_time = orders_time_per_building_type.value[i]
        order_hours_df.loc[len(order_hours_df)] = [
            lib.get_building_name(owned_buildings_idxs[i]),
            orders_time / buildings_vec_subset[i] * 24
        ]
    print(order_hours_df)
    print("\n\n")

    orders_df = pd.DataFrame(columns=["Building", "Recipe", "Num orders"])

    for i in range(orders_var.value.size):
        recipe_id = possible_recipe_idxs[i]
        if orders_var.value[i] > 0:
            orders_df.loc[len(orders_df)] = [
                lib.GAME_DATA["buildings"][lib.GAME_DATA["recipes"][recipe_id-1]["producedIn"]-1]["name"],
                lib.get_recipe_output_name(recipe_id),
                int(orders_var.value[i] + EPSILON)
            ]
    orders_df = orders_df.sort_values(["Building", "Recipe"])

    print(orders_df)
    print("\n\n")

    mats_df = pd.DataFrame(columns=["Material", "At Base Warehouse", "From Exchange Warehouse", "Purchase", "Final"])

    result_purchases = (purchases_var.value.sum(1) + EPSILON).astype(int)
    result_costs = ((purchases_var.value * purchase_costs).sum(1) + EPSILON).astype(int) / 100
    result_weights = result_purchases * weights_vec[possible_mats_ids]
    purchases_var_value = np.array(purchases_var.value)
    resulting_mats_value = np.array(resulting_mats.value)
    for i in range(resulting_mats_value.size):
        if total_storage_vec_subset[i] != resulting_mats_value[i] or result_purchases[i] - EPSILON > 0:
            mats_df.loc[len(mats_df)] = [
                lib.render_mat_with_icon(lib.get_material_name(possible_mats_ids[i]), link=True),
                int(base_warehouse_vec_subset[i] + lib.EPSILON),
                0.0,
                result_purchases[i],
                int(resulting_mats_value[i] + lib.EPSILON)
            ]
    mats_df = mats_df.sort_values("Purchase", ascending=False)
    print(mats_df)
    print("\n\n")

    purchases_df = pd.DataFrame(columns=["Material", "To Purchase", "Cost", "Weight"])
    wishlist = []
    for i in range(result_purchases.size):
        if result_purchases[i] - EPSILON > 0:
            purchases_df.loc[len(purchases_df)] = [
                lib.render_mat_with_icon(lib.get_material_name(possible_mats_ids[i]), link=True),
                result_purchases[i],
                result_costs[i],
                result_weights[i]
            ]
            wishlist.append({"id": int(possible_mats_ids[i]), "am": int(result_purchases[i])})
    purchases_df = purchases_df.sort_values("Cost", ascending=False)    

    return purchases_df, order_hours_df, orders_df, mats_df, wishlist



def plan_production(base_id,
                    bases_data,
                    warehouses_data,
                    budget,
                    period,
                    exchange_data,
                    consumables_duration=None,
                    max_weight=None,
                    include_consumables=True,
                    include_exchange=True,
                    no_inputs_sellable=True,
                    only_outputs_sellable=True,
                    no_outputs_purchaseable=True):
    base_data = bases_data[base_id]    
    recipe_to_mats_matrix = lib.get_recipe_to_mats_matrix()
    recipe_time_vec = lib.get_recipe_time_vec()
    building_to_recipe_matrix = lib.get_building_to_recipe_matrix()
    base_warehouse_vec = lib.get_base_storage_vec(base_data, warehouses_data)
    exchange_warehouse_vec = lib.get_warehouse_contents_vec(lib.get_exchange_warehouse_data(warehouses_data))

    exchange_orders_data = lib.load_exchange_orders()
    other_warehouses_mat_vec = lib.get_company_wide_mats(warehouses_data, exchange_orders_data)
    other_warehouses_mat_vec -= base_warehouse_vec

    # storage_vec[lib.get_mat_id("Wood")] = 20000
    # storage_vec[lib.get_mat_id("Carbon")] = 0
    buildings_vec = lib.get_building_levels_vec(base_data)
    weights_vec = lib.get_weight_vec()
    daily_consumption_vec = lib.get_daily_consumption_vec(base_data)

    owned_buildings_idxs = np.intersect1d(np.where(buildings_vec)[0], lib.get_production_buildings(building_to_recipe_matrix))
    buildings_vec_subset = buildings_vec[owned_buildings_idxs]

    possible_recipe_idxs = np.where(np.matmul(buildings_vec_subset, building_to_recipe_matrix[owned_buildings_idxs]))[0]

    buildings_to_recipe_matrix_subset = building_to_recipe_matrix[owned_buildings_idxs][:, possible_recipe_idxs]
    recipe_to_mats_matrix_subset = recipe_to_mats_matrix[possible_recipe_idxs]
    recipe_time_vec_subset = recipe_time_vec[possible_recipe_idxs]

    possible_mats_ids = np.where(np.logical_or((recipe_to_mats_matrix_subset != 0).any(0), daily_consumption_vec != 0))[0]
    recipe_to_mats_matrix_subset = recipe_to_mats_matrix_subset[:, possible_mats_ids]
    base_warehouse_vec_subset = base_warehouse_vec[possible_mats_ids]
    exchange_warehouse_vec_subset = exchange_warehouse_vec[possible_mats_ids]
    daily_consumption_vec_subset = daily_consumption_vec[possible_mats_ids]
    weights_vec_subset = weights_vec[possible_mats_ids]
    other_warehouses_mat_vec_subset = other_warehouses_mat_vec[possible_mats_ids]

    max_sell_orders = 0
    for mat_id in possible_mats_ids:
        max_sell_orders = max(max_sell_orders, len(exchange_data[mat_id]["orders"]))

    # Variables

    orders_var = cp.Variable(possible_recipe_idxs.shape[0], integer=True)
    purchases_var = cp.Variable([len(possible_mats_ids), max_sell_orders], integer=True)
    
    purchase_costs = np.zeros(purchases_var.shape)
    purchase_qtys = np.zeros(purchases_var.shape)
    for i, mat_id in enumerate(possible_mats_ids):
        for j, sell_order in enumerate(exchange_data[mat_id]["orders"]):
            purchase_costs[i, j] = sell_order["unitPrice"]
            purchase_qtys[i, j] = sell_order["qty"]
    prices_vec_subset = purchase_costs[:, 0]

    NUM_BUCKETS = 10
    sales_var = cp.Variable([len(possible_mats_ids), NUM_BUCKETS], integer=True)
    sales_prices = np.zeros(sales_var.shape)
    sales_max_amts = np.zeros(sales_var.shape)
    for i, mat_id in enumerate(possible_mats_ids):
        sales_max_amts[i] = exchange_data[mat_id]["avgQtySoldDaily"] / NUM_BUCKETS * 2 #/ 2 # Divide by 2 is arbitrary
        for j in range(sales_prices.shape[1]):
            sales_prices[i, j] = exchange_data[mat_id]["currentPrice"] * (0.8 + ((j + 1) * 0.025))
    sales_revenue = cp.sum(cp.multiply(sales_var, sales_prices))

    input_mat_mask = (recipe_to_mats_matrix_subset < 0).any(0)
    output_mat_mask = (recipe_to_mats_matrix_subset > 0).any(0)
    if no_inputs_sellable:
        sales_prices *= np.expand_dims(~(np.logical_and(input_mat_mask, ~output_mat_mask)), axis=1)
    # if no_outputs_purchaseable:
    #     purchase_qtys *= np.expand_dims(~output_mat_mask, axis=1)

    def get_possible_idx(name):
        return np.where(possible_mats_ids == lib.get_mat_id(name))[0][0]

    # sales_prices[get_possible_idx("Ethanol")] *= 100
    # prices_vec_subset[get_possible_idx("Biopolyne")] *= 3
    # purchase_qtys[get_possible_idx("Biopolyne")] = 0


    total_storage_vec_subset = base_warehouse_vec_subset
    # if include_exchange:
    #     total_storage_vec_subset = total_storage_vec_subset + exchange_warehouse_vec_subset

    resulting_mats = (cp.matmul(orders_var, recipe_to_mats_matrix_subset)) + total_storage_vec_subset + purchases_var.sum(1)
    if consumables_duration is None:
        consumables_duration = period
    if include_consumables:
        needs_consumbles = (total_storage_vec_subset - daily_consumption_vec_subset * period < 0).any()
        if needs_consumbles:
            resulting_mats = resulting_mats - daily_consumption_vec_subset * consumables_duration

    orders_time = cp.multiply(orders_var, recipe_time_vec_subset)
    orders_time_per_building_type = cp.matmul(orders_time, buildings_to_recipe_matrix_subset.T)

    total_purchase_cost = cp.sum(cp.multiply(purchases_var, purchase_costs))

    total_value = cp.sum(cp.multiply(resulting_mats, prices_vec_subset))

    # objective = cp.Maximize(cp.sum(total_value) - cp.sum(total_purchase_cost) * 1.1)
    objective = cp.Maximize(sales_revenue - cp.sum(total_purchase_cost) * 1.1)

    constraints = [
        # Orders are non-negative
        orders_var >= 0,
        # Goods are non-negative after production
        resulting_mats >= 0,
        # Goods after sales is non-negative
        resulting_mats + other_warehouses_mat_vec_subset - cp.sum(sales_var, 1) >= 0,
        # Purchases are non-negative
        purchases_var >= 0,
        # Purchases come from existing sell orders
        purchases_var <= purchase_qtys,
        # Orders match available building production levels
        orders_time_per_building_type / buildings_vec_subset <= period,
        # Sales limit
        sales_var <= sales_max_amts,
        # Sales are non-negative
        sales_var >= 0,
        # # No purchased goods are sold
        # cp.sum(cp.multiply(purchases_var.sum(1), sales_var.sum(1))) == 0

        # Only researched recipes used
        # cp.sum(cp.multiply(orders_var, (1 - recipe_is_doable.astype(float)))) == 0,
        # cp.sum(cp.multiply(lib.get_weight_vec(), purchases_var)) <= 6000
    ]


    # Purchases are within budget
    if budget is not None and budget >= 0:
        constraints.append(total_purchase_cost <= budget * 100)
    # breakpoint()

    if max_weight is not None:
        constraints.append(cp.sum(cp.multiply(purchases_var.sum(1), weights_vec_subset)) <= max_weight)

    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=True)
    # result = prob.solve()

    order_hours_df = pd.DataFrame(columns=["Building", "Hours"])
    for i in range(buildings_vec_subset.shape[0]):
        orders_time = orders_time_per_building_type.value[i]
        order_hours_df.loc[len(order_hours_df)] = [
            lib.get_building_name(owned_buildings_idxs[i]),
            orders_time / buildings_vec_subset[i] * 24
        ]
    print(order_hours_df)
    print("\n\n")

    orders_df = pd.DataFrame(columns=["Building", "Recipe", "Num orders"])

    for i in range(orders_var.value.size):
        recipe_id = possible_recipe_idxs[i]
        if orders_var.value[i] > 0:
            orders_df.loc[len(orders_df)] = [
                lib.GAME_DATA["buildings"][lib.GAME_DATA["recipes"][recipe_id-1]["producedIn"]-1]["name"],
                lib.get_recipe_output_name(recipe_id),
                int(orders_var.value[i] + EPSILON)
            ]
    orders_df = orders_df.sort_values(["Building", "Recipe"])

    print(orders_df)
    print("\n\n")

    mats_df = pd.DataFrame(columns=["Material", "At Base Warehouse", "From Exchange Warehouse", "Purchase", "Final"])

    result_purchases = (purchases_var.value.sum(1) + EPSILON).astype(int)
    result_costs = ((purchases_var.value * purchase_costs).sum(1) + EPSILON).astype(int) / 100
    result_weights = result_purchases * weights_vec[possible_mats_ids]
    purchases_var_value = np.array(purchases_var.value)
    resulting_mats_value = np.array(resulting_mats.value)
    for i in range(resulting_mats_value.size):
        if total_storage_vec_subset[i] != resulting_mats_value[i] or result_purchases[i] - EPSILON > 0:
            mats_df.loc[len(mats_df)] = [
                lib.render_mat_with_icon(lib.get_material_name(possible_mats_ids[i]), link=True),
                int(base_warehouse_vec_subset[i] + lib.EPSILON),
                0.0,
                result_purchases[i],
                int(resulting_mats_value[i] + lib.EPSILON)
            ]
    mats_df = mats_df.sort_values("Purchase", ascending=False)
    print(mats_df)
    print("\n\n")

    purchases_df = pd.DataFrame(columns=["Material", "To Purchase", "Cost", "Weight"])
    wishlist = []
    for i in range(result_purchases.size):
        if result_purchases[i] - EPSILON > 0:
            purchases_df.loc[len(purchases_df)] = [
                lib.render_mat_with_icon(lib.get_material_name(possible_mats_ids[i]), link=True),
                result_purchases[i],
                result_costs[i],
                result_weights[i]
            ]
            wishlist.append({"id": int(possible_mats_ids[i]), "am": int(result_purchases[i])})
    purchases_df = purchases_df.sort_values("Cost", ascending=False)    

    return purchases_df, order_hours_df, orders_df, mats_df, wishlist


def get_full_production_time(base_id, bases_data, warehouses_data, limit_by_prod_orders=False):
    base_data = bases_data[base_id]
    recipe_to_mats_matrix = lib.get_recipe_to_mats_matrix()
    recipe_time_vec = lib.get_recipe_time_vec()
    building_to_recipe_matrix = lib.get_building_to_recipe_matrix()
    storage_vec = lib.get_base_storage_vec(base_data, warehouses_data)
    buildings_vec = lib.get_building_levels_vec(base_data)
    prices_vec = lib.load_prices_vec()
   
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
    print(base_data["name"])

    result = prob.solve()

    order_hours_df = pd.DataFrame(columns=["Building", "Hours"])
    for i in range(buildings_vec_subset.shape[0]):
        orders_time = orders_time_per_building_type.value[i]
        order_hours_df.loc[len(order_hours_df)] = [
            lib.get_building_name(owned_buildings_idxs[i]),
            orders_time / buildings_vec_subset[i] * 24
        ]
    print(order_hours_df)
    print("\n\n")

    orders_df = pd.DataFrame(columns=["Building", "Recipe", "Num orders"])

    for i in range(orders_var.value.size):
        recipe_id = possible_recipe_idxs[i]
        if orders_var.value[i] > 0:
            orders_df.loc[len(orders_df)] = [
                lib.GAME_DATA["buildings"][lib.GAME_DATA["recipes"][recipe_id-1]["producedIn"]-1]["name"],
                lib.get_recipe_output_name(recipe_id),
                int(orders_var.value[i])
            ]

    print(orders_df)
    print("\n\n")
    # breakpoint()
    return (orders_time_per_building_type.value / buildings_vec_subset).min()

    # return result