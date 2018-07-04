from django.http import HttpResponse
# Create your views here.

from django.shortcuts import render
# !/usr/bin/python
# -*- coding: utf-8 -*-
from operator import attrgetter
from collections import namedtuple
from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from django.http import JsonResponse
import json
from optimizer.models import ScheduledActivity
from django.template.defaulttags import register
import pandas as ps
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import ColumnDataSource, Range1d, DatetimeTickFormatter, FixedTicker
from bokeh.models.tools import HoverTool
import datetime
from bokeh.io import export_png
from bokeh.plotting import output_file, save


@register.filter
def get_task_status(status):
    if status == '0':
        return 'Pending'
    elif status == '1':
        return 'Progress'
    else:
        return 'Completed'


def index(request):
    # prod_orders = generateschduele()
    return render(request, 'index.html', {'schedule': 'schedule.png', 'orders': ScheduledActivity.objects.all()})


def getschedule(request):
    print('API Called!!')
    if request.FILES:
        filename = storefile(request.FILES['file'])
        prod_orders = generateschduele(filename)
    else:
        prod_orders = generateschduele()
    _deletestoredactivities()
    _storeactivities(prod_orders)
    data = {'schedule_image': _baseurl(request)['BASE_URL'] + "/static/schedule.png",
            'next_activity': _getnextscheduledactivity()}
    return JsonResponse(data, safe=False)


def storefile(f):
    with open('data/PolishingOrder.xlsx', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return 'data/PolishingOrder.xlsx'


def getnextactivity(request):
    ScheduledActivity.objects.filter(status=1).update(status=2)
    act = ScheduledActivity.objects.filter(status=0).values()[0]
    ScheduledActivity.objects.filter(id=act['id']).update(status=1)
    data = {'next_activity': act}
    return JsonResponse(data, safe=False)


def _baseurl(request):
    """
    Return a BASE_URL template context for the current request.
    """
    if request.is_secure():
        scheme = 'https://'
    else:
        scheme = 'http://'

    return {'BASE_URL': scheme + request.get_host(), }


def _storeactivities(orders):
    for order in orders:
        sa = ScheduledActivity(order_id=order['order_id'], product_id=order['product_id'],
                               product_name=order['product_name'], quantity=order['quantity'],
                               start_datetime=order['start_date'], end_datetime=order['end_date'], activitytype=0,
                               status=0)
        sa.save()


def _getnextscheduledactivity():
    act = ScheduledActivity.objects.filter(status=0).values()[0]
    ScheduledActivity.objects.filter(id=act['id']).update(status=1)
    return act


def _deletestoredactivities():
    ScheduledActivity.objects.all().delete()


Item = namedtuple("Item", ['index', 'requiredTime', 'deadline'])
Product = namedtuple('Product', ['index', 'name', 'setupTime', 'unitProductionTime'])
Order_new = namedtuple('Order_new', ['orderID', 'productIndex', 'quantity', 'productcode', 'deadline', 'workIndex'])
MaintenanceDate_start = 5000
MaintenanceDate_end = 6000


def solve(input_data, capacity=None):
    # Modify this code to run your optimization algorithm

    # parse the input
    deadlines = [data[2] for data in input_data]
    requiredTimes = [data[1] for data in input_data]
    if capacity is None:
        capacity = max(deadlines)
        # capacity = sum(requiredTimes)

    [value, taken] = ProcessOrder(deadlines, requiredTimes, capacity)
    # prepare the solution in the specified output format

    combined_items = []
    for i, item in enumerate(taken):
        taken_order = []
        if item == 1:
            combined_items.append(Item(i, requiredTimes[i], deadlines[i]))
    # Fix Scheduling time
    # Sort as per deadline
    # 1. First endDate set to time duration of that order
    # 2. Add next time duration and so on
    sorted_deadlines = sorted(combined_items, key=attrgetter('deadline'))
    print("RESULTS:")
    output_data = str(value) + ' ' + str(0)
    # output_data += ' '.join(map(str, taken))
    print(output_data)
    data_for_graph = []
    for i, item in enumerate(sorted_deadlines):
        if i == 0:
            data_for_graph.append([item.index, 0,
                                   item.requiredTime])
        else:
            data_for_graph.append([item.index, data_for_graph[i - 1][-1],
                                   data_for_graph[i - 1][-1] + item.requiredTime])
    # plot_schedule(data_for_graph)
    return taken


def zeros(rows, cols):
    zeroList = []
    for i in range(0, rows):
        tmpList = []
        for j in range(0, cols):
            tmpList.append(0)
        zeroList.append(tmpList)
    return zeroList


def getUsedItems(w, c):
    i = len(c) - 1
    currentW = len(c[0]) - 1
    marked = []
    for i in range(i + 1):
        marked.append(0)
    while (i >= 0 and currentW >= 0):
        if (i == 0 and c[i][currentW] > 0) or c[i][currentW] != c[i - 1][currentW]:
            marked[i] = 1
            currentW = currentW - w[i]
        i = i - 1
    return marked


def ProcessOrder(d, r, K):
    # Maximize Machine Usage
    n = len(d)
    c = zeros(n, K + 1)
    for i in range(n):  # i in [0, 1] if there are two items
        for j in range(K + 1):  # j in [0, 1, 2...11] if deadline is 11
            if r[i] <= j:
                c[i][j] = max(r[i] + c[i - 1][j - r[i]], c[i - 1][j])
            else:
                c[i][j] = c[i - 1][j]
    return [c[n - 1][K], getUsedItems(r, c)]


def ProcessProdcut(d, r, s, K):
    # Minimize Machine Setup
    # d =
    # r=
    # K=Capacity
    n = len(d)  # number of orders
    c = zeros(n, K + 1)
    for i in range(n):  # i in [0, 1] if there are two items
        for j in range(K + 1):  # j in [0, 1, 2...11] if deadline is 11
            if r[i] <= j:
                c[i][j] = min(r[i] + c[i - 1][j - r[i]], c[i - 1][j])
            else:
                c[i][j] = c[i - 1][j]
    return [c[n - 1][K], getUsedItems(r, c)]


colors = ["r", "g", "b", "y", "c", "m", '#FF9900', '#5d8984', '#c00000', '#672483', '#77933c',
          '#ffb445', '#89a9a5', '#d14545', '#905fa4', '#9cb071',
          '#a36200', '#3c5853', '#7b0000', '#421754', '#4c5e27',
          '#ffd08b', '#b5c9c7', '#e28b8b', '#b99bc6', '#c1cda6',
          '#5d3800', '#223230', '#460000', '#260e30', '#2c3616',
          ]


def get_color_pallet(size):
    if size < 65:
        return colors[:size]
    p = colors
    out = []
    for i in range(size):
        idx = int(i * 25.0 / size)
        out.append(p[idx])
    return out


def get_clean_data(order_data, product_data):
    product_lines = product_data.split('\n')
    product_count = len(product_lines) - 1
    products = []
    for i in range(1, product_count):
        lines = product_lines[i].split(',')
        products.append(Product(int(lines[3]), lines[0], int(lines[1]), int(lines[2])))

    order_lines = order_data.split('\n')
    order_count = len(order_lines) - 1
    orders = []
    for i in range(1, order_count):
        lines = order_lines[i].split(',')
        orders.append(Order_new(int(lines[0]), int(lines[1]), int(lines[2]),
                                int(lines[3]), int(lines[4])))
    processed_orders = []
    for order in orders:
        # in format Order(requiredTime, deadline)
        product = [p for p in products if p.index == order.productIndex]
        processed_orders.append([order.orderID, int(order.quantity / product[0].unitProductionTime
                                                    + product[0].setupTime),
                                 order.deadline, order[1]])

        # Combining order wise
    combined_orders = []
    for orderID in set([order[0] for order in processed_orders]):
        combined_orders.append([orderID,
                                sum([order[1] for order in processed_orders if order[0] == orderID]),  # Required Time
                                ([order[2] for order in processed_orders if order[0] == orderID])[0],  # Deadlines
                                ])

    return products, orders, processed_orders, combined_orders


def getProductWithMaxSetupTime(common, products):
    if not common:
        common = {0}
    max_value = max([product.setupTime for product in products if product.index in common])
    productID = [product.index for product in products if product.setupTime == max_value and product.index in common]
    return productID[0]


def getProductWithMinSetupTime(remaining, products):
    max_value = max([product.setupTime for product in products if product.index in remaining])
    productID = [product.index for product in products if product.setupTime == max_value and product.index in remaining]
    return productID[0]


def optimize_product(taken_products, products):
    # get Order Id in sequence
    orderIDs = list(OrderedDict.fromkeys([order[0] for order in taken_products]))
    # Find Common Product between Order1 and following Order 2
    common_products = []
    for i, orderID in enumerate(orderIDs):
        if i < (len(orderIDs) - 1):
            current_products = set([taken[3] for taken in taken_products if taken[0] == orderID])
            next_products = set([taken[3] for taken in taken_products if taken[0] == orderIDs[i + 1]])
            common_products.append(current_products & next_products)
    common_products = list(common_products)
    print('\nCommon Products')
    print(common_products)

    # 2. find product of maximum setUpTime
    transitions = []
    last_item = None
    for i, common in enumerate(common_products):
        if i != 0:
            last_item = transitions[i - 1]
        maxProductID = getProductWithMaxSetupTime(common, products)
        if [maxProductID] == last_item:
            transitions.append([])
        else:
            transitions.append([maxProductID])

    # Find Product with minimum setUpTime from remaining productID for every
    # order
    print('\nTransitions')
    print(transitions)
    orders = taken_products
    otherProducts = []
    for i, orderID in enumerate(orderIDs):
        if i == 0:
            otherProducts.append(
                [order[3] for order in orders if order[0] == orderID and order[3] not in transitions[i]])
        elif i == (len(orderIDs) - 1):
            otherProducts.append(
                [order[3] for order in orders if order[0] == orderID and order[3] not in transitions[i - 1]])
        else:
            otherProducts.append([order[3] for order in orders if
                                  order[0] == orderID and order[3] not in transitions[i] and order[3] not in
                                  transitions[i - 1]])
    slots = []
    combined_sequence = []
    for i, other in enumerate(otherProducts):
        if i == 0:
            try:
                other.append(transitions[i][0])
            except:
                pass
        elif i == (len(otherProducts) - 1):
            try:
                other.insert(0, transitions[i - 1][0])
            except:
                pass
        else:
            try:
                other.insert(0, transitions[i - 1][0])
            except:
                pass
            try:
                other.append(transitions[i][0])
            except:
                pass
        combined_sequence.append(other)
    return orderIDs, combined_sequence


def load_data_from_file(file_name):
    df = pd.read_excel(io=file_name)
    return df


def gettimestamp(date):
    import time
    import datetime
    ref_date_str = "4/26/2018 8:00"
    ref_date = datetime.datetime.strptime(ref_date_str, "%m/%d/%Y %H:%M")
    m, s = divmod((date - ref_date).seconds, 60)
    return m * 100


def getstarttimestamp():
    import time
    import datetime
    ref_date_str = "4/26/2018 8:00"
    return int(time.mktime(datetime.datetime.strptime(ref_date_str, "%m/%d/%Y %H:%M").timetuple()))


def timestamptodatestring(timestamp):
    import datetime
    return datetime.datetime.fromtimestamp(
        int(getstarttimestamp() + timestamp)
    ).strftime('%Y-%m-%d %H:%M')


def get_data(order_data_df, product_data):
    product_lines = product_data.split('\n')
    product_count = len(product_lines) - 1
    products = []
    for i in range(1, product_count):
        lines = product_lines[i].split(',')
        products.append(Product(int(lines[3]), lines[0], int(lines[1]), float(lines[2])))

    # order_lines = order_data.split('\n')
    order_count = len(order_data_df.index)

    orders = []
    orders_map = {}
    for index, row in order_data_df.iterrows():
        orders_map.update(
            {row['Production Order Nr.']: [row['Name of product'], row['Quantity'], row['code of product']]})
        orders.append(
            Order_new(row['Production Order Nr.'], row['ProductType'], row['Quantity'], row['code of product'],
                      int(gettimestamp(row['End Time'])),
                      index))

    # for i in range(1, order_count):
    #     lines = order_lines[i].split(',')
    #     orders.append(Order(int(lines[0]), int(lines[1]), int(lines[2]),
    #                         int(lines[3]), int(lines[4])))
    processed_orders = []
    for order in orders:
        # in format Order(requiredTime, deadline)
        product = [p for p in products if p.index == order.productIndex]
        processed_orders.append([order.orderID, int(order.quantity * product[0].unitProductionTime
                                                    + product[0].setupTime),
                                 order.deadline, order[1]])

        # Combining order wise
    combined_orders = []
    for orderID in set([order[0] for order in processed_orders]):
        combined_orders.append([orderID,
                                sum([order[1] for order in processed_orders if order[0] == orderID]),  # Required Time
                                ([order[2] for order in processed_orders if order[0] == orderID])[0],  # Deadlines
                                ])
    return products, orders, processed_orders, combined_orders, orders_map


def generateschduele(file_location="data/Polishing Production Orders JUNHO2017_20180409.xlsx"):
    # TODO provide filename
    # file_location = "data/Polishing Production Orders JUNHO2017_20180409.xlsx"
    capacity = None

    # load_data_from_file(file_location)
    # with open(file_location, 'r') as input_data_file:
    #     input_data = input_data_file.read()
    with open("data/products.txt", 'r') as product_file:
        product_data = product_file.read()
    products, orders, processed_orders, combined_orders, orders_map = get_data(load_data_from_file(file_location),
                                                                               product_data)
    if capacity is None:
        taken = solve(combined_orders)
    else:
        taken = solve(combined_orders, capacity=capacity)
    taken_products = [order for order in processed_orders if taken[order[3]] == 1]
    # taken_products=[]
    # for order in processed_orders:
    #     import ipdb
    #     ipdb.set_trace()
    #     if taken[order[0]]==1:
    #         taken_products.append(order)
    orderIDs, combined_sequence = optimize_product(taken_products, products)
    print("\nTAKEN ORDERS")
    print(taken_products)
    print("\nWe should run following ORDERS: ")
    print(orderIDs)
    print("\nPRODUCT Run Sequence- Combined Sequence")
    for item in combined_sequence:
        print(item)
    new_list = []
    # Re-arrage taken orders
    for i, orderID in enumerate(orderIDs):
        new_list.append([order for order in taken_products if order[0] == orderID])
    # Sort new_list as per combined_sequence
    print("\nNew LIST")
    for item in new_list:
        print(item)
    final_list = []
    for i, item in enumerate(new_list):
        final_list.append(sorted(item, key=lambda x: combined_sequence[i].index(x[3])))

    print("\nFinal LIST")
    for item in final_list:
        print(item)
    return generatescheduledata(final_list, products, orders_map)


def generatescheduledata(list_process_orders, products, orders_map):
    left = 0
    color_index = -1
    start = 0
    final_orders = []
    for j, process_orders in enumerate(list_process_orders):
        order_width = 0
        order_index = -1
        for i, order in enumerate(process_orders):
            order_index = order[0]
            value = order[1]
            if i == 0:
                start = left
            if color_index != -1:
                setupTime = [(product.setupTime) + 15 for product in products if product.index == color_index]
            else:
                setupTime = [(product.setupTime) + 90 for product in products if product.index == order[3]]
            setupTime = setupTime[0]
            # if color_index != order[3]:
            value = order[1] + setupTime
            left += setupTime
            color_index = order[3]
            # x_ticks.append(left)
            if MaintenanceDate_start <= left + value and left <= MaintenanceDate_end:
                # p_order = {'order_id': 'NA', 'product_name': 'MAINTENANCE',
                #            'quantity': 0, 'product_id': 'NA',
                #            'start_date': timestamptodatestring(MaintenanceDate_start),
                #            'end_date': timestamptodatestring(MaintenanceDate_end)}
                # final_orders.append(p_order)
                left = MaintenanceDate_end
            left += value
        order_width = left - start
        p_order = {'order_id': order_index, 'product_name': orders_map[order_index][0],
                   'quantity': orders_map[order_index][1], 'product_id': orders_map[order_index][2],
                   'start_date': timestamptodatestring(start + setupTime),
                   'end_date': timestamptodatestring(left)}
        final_orders.append(p_order)
    # final_orders.append({'order_id': '', 'product_name': 'Maintenance',
    #                'start_date': timestamptodatestring(MaintenanceDate_start),
    #                'end_date': timestamptodatestring(MaintenanceDate_end)})
    generate_schedule_graph(final_orders)
    return final_orders


def date_range(start_date, end_date, increment, period):
    from dateutil.relativedelta import relativedelta
    result = []
    nxt = start_date
    delta = relativedelta(**{period: increment})
    while nxt <= end_date:
        result.append(str(nxt))
        nxt += delta
    return result


def generate_schedule_graph(final_orders):
    DF = ps.DataFrame(columns=['Item', 'Start', 'End', 'Color'])
    items = []
    for order in final_orders:
        l = [str(order['order_id']) + '-' + order['product_name'], order['start_date'], order['end_date']]
        l.append('Green')
        items.append(l)
    for i, Dat in enumerate(items[::-1]):
        DF.loc[i] = Dat
    DF['Start_dt'] = ps.to_datetime(DF.Start)
    DF['End_dt'] = ps.to_datetime(DF.End)
    G = figure(title='Polishing Schedule', x_axis_type='datetime', width=1200, height=400, y_range=DF.Item.tolist(),
               x_range=Range1d(DF.Start_dt.min(), DF.End_dt.max(), min_interval=datetime.timedelta(minutes=30)))
    G.xaxis.formatter = DatetimeTickFormatter(
        hours=["%d %B %Y %H:%m"],
        days=["%d %B %Y %H:%m"],
        months=["%d %B %Y %H:%m"],
        years=["%d %B %Y %H:%m"],
    )
    tick_vals = pd.to_datetime(date_range(DF.Start_dt.min(), DF.End_dt.max(), 2, 'hours')).astype(int) / 10 ** 6
    G.xaxis.ticker = FixedTicker(ticks=list(tick_vals))
    hover = HoverTool(tooltips="Product: @Item<br>\
    Start: @Start<br>\
    End: @End")
    G.add_tools(hover)

    DF['ID'] = DF.index + 0.8
    DF['ID1'] = DF.index + 1.2
    CDS = ColumnDataSource(DF)
    G.quad(left='Start_dt', right='End_dt', bottom='ID', top='ID1', source=CDS, color="Color")
    # G.rect(,"Item",source=CDS)
    # show(G)
    export_png(G, filename="static/schedule.png")
    output_file('static/schedule.html', mode='inline')
    save(G)


def plot_schedule(list_process_orders, products, orders_map):
    import matplotlib.patches as mpatches

    # make sure colors is greater than number of products
    colors = get_color_pallet(50)
    hatches = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')

    values = np.array(list_process_orders)
    fig = plt.figure(figsize=(20, 5), dpi=80, facecolor='w', edgecolor='k')
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(211)
    left = 0
    color_index = -1
    start = 0
    x_ticks = [0]
    y_ticks = [0]
    top = 1
    ax1.axvline(x=left, color='black')
    ax1.axvline(x=MaintenanceDate_start, color='red')
    ax1.axvline(x=MaintenanceDate_end, color='red')
    final_orders = []
    for j, process_orders in enumerate(list_process_orders):
        order_width = 0
        order_index = -1
        for i, order in enumerate(process_orders):
            order_index = order[0]
            value = order[1]
            if i == 0:
                start = left
            if color_index != -1:
                setupTime = [(product.setupTime) + 15 for product in products if product.index == color_index]
            else:
                setupTime = [(product.setupTime) + 90 for product in products if product.index == order[3]]
            setupTime = setupTime[0]
            # if color_index != order[3]:
            value = order[1] + setupTime
            ax1.axvline(x=left, color='yellow', alpha=0.4)
            x_ticks.append(left)
            left += setupTime
            ax1.axvline(x=left, color='yellow', alpha=0.4)
            x_ticks.append(left)
            color_index = order[3]
            # x_ticks.append(left)
            if MaintenanceDate_start <= left + value and left <= MaintenanceDate_end:
                left = MaintenanceDate_end

            ax1.barh(y=top, left=left, width=value, linewidth=0.5,
                     color=colors[color_index], tick_label='test')

            left += value
            x_ticks.extend(list(range(value, left)))
            top += 1

            y_ticks.append(top)
        ax1.text((start + left) / 2, 0, 'O_%i' % order_index, size=8, ha='center')
        ax1.axvline(x=left, color='black')
        order_width = left - start
        p_order = {'order_id': order_index, 'product_name': orders_map[order_index],
                   'start_date': timestamptodatestring(start + setupTime),
                   'end_date': timestamptodatestring(left)}
        ax1.barh(y=0, left=start, width=order_width, linewidth=1,
                 label='O_%i' % order_index)
        final_orders.append(p_order)
    new_x_ticks = x_ticks
    ax1.set_xticks(list(set(new_x_ticks))[::200])
    ax1.set_xticklabels(list(set(new_x_ticks))[::200], rotation=90)
    # ax2.set_xticks(new_x_ticks)

    ax1.set_yticks([])
    # ax2.set_yticks([])

    ax1.set_xlim(xmin=0)
    # ax2.set_xlim(xmin=0)

    # ax2.set_xlabel("Orders with time")
    ax1.set_ylabel("Machine 0")
    # ax2.set_ylabel("Machine 0")

    # set titles
    ax1.set_title('Polishing Schedule')
    # ax2.set_title('Orders schedule')

    # Legend For Colors
    patch_list = []

    # for order, color in zip(list_process_orders, colors):
    #     data_key = mpatches.Patch(color=color, label= "Order_%i" %order[0][0])
    #     patch_list.append(data_key)
    for product, color in zip(products, colors):
        data_key = mpatches.Patch(color=color, label=product.name)
        patch_list.append(data_key)
    ax1.legend(handles=patch_list, loc="best", bbox_to_anchor=(1.0, 1.00))

    # ax2.legend(loc="best", bbox_to_anchor=(1.0, 1.00))
    plt.tick_params(axis='both', labelsize=6)
    # plt.show()
    fig.savefig('static/schedule.png')  # save the figure to file
    # plt.close(fig)

    return final_orders


def maintainprocessedhistory():
    print('history maintenance')
