
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
from django.db.models import Q
import datetime
from math import pi

completed_orders = []
working_hrs = [datetime.time(8, 00), datetime.time(12, 00), datetime.time(13, 30), datetime.time(17, 30)]
def read_productiontime():
    df = pd.read_excel(io='data/productiontime_20180730.xlsx')
    prod_time = {}
    for index, row in df.iterrows():
        prod_time[row['product type identifier'].split('-')[1] + str(int(row['Quality']))] = (float(
            row['production time']) * 3600 / 3000) / 10
    return prod_time

def read_actual_productiontime():
    df = pd.read_excel(io='data/productiontime_20180730.xlsx')
    prod_time = {}
    for index, row in df.iterrows():
        prod_time[row['product type identifier'].split('-')[1] + str(int(row['Quality']))] = (float(
            row['production time'])/3000)
    return prod_time

prod_time = read_productiontime()

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
    all_orders = ScheduledActivity.objects.values()
    incomplete_order = ScheduledActivity.objects.filter(~Q(status=2)).values()
    generate_schedule_graph(all_orders)
    return render(request, 'index.html', {'schedule': 'schedule.png', 'orders': all_orders})


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


from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def mark_completed(request):
    order_id = request.POST.get('order_id')
    _collect_completed_orders()
    if not order_id:
        return JsonResponse({'error_msg': 'order id not provided'}, safe=False)
    ScheduledActivity.objects.filter(order_id=order_id).update(status=2)
    _deleteincompleteschedules()
    prod_orders = generateschduele()
    _storeactivities(prod_orders)
    data = {'schedule_image': _baseurl(request)['BASE_URL'] + "/static/schedule.png",
            'next_activity': _getnextscheduledactivity()}
    return JsonResponse(data, safe=False)


def _collect_completed_orders():
    global completed_orders
    completed_orders = ScheduledActivity.objects.filter(status=2).values_list('order_id', flat=True)


def storefile(f):
    with open('data/PolishingOrders_new.xlsx', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return 'data/PolishingOrders_new.xlsx'


def getnextactivity(request):
    ScheduledActivity.objects.filter(status=1).update(status=2)
    act = ScheduledActivity.objects.filter(status=0).values()
    if act:
        n_act = act[0]
        ScheduledActivity.objects.filter(id=n_act['id']).update(status=1)
    if act:
        data = {'next_activity': n_act}
    else:
        data = {'next_activity': None}
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
        sa = ScheduledActivity(order_id=order['order_id'], product_id=order['product_id'].strip(),
                               product_name=order['product_name'], quantity=order['quantity'],
                               start_datetime=order['start_datetime'], end_datetime=order['end_datetime'],
                               activitytype=0,
                               status=0)
        sa.save()


def _getnextscheduledactivity():
    act = ScheduledActivity.objects.filter(status=0).values()[0]
    ScheduledActivity.objects.filter(id=act['id']).update(status=1)
    return act


def _deletestoredactivities():
    ScheduledActivity.objects.all().delete()


def _deleteincompleteschedules():
    ScheduledActivity.objects.filter(Q(status=0) | Q(status=1)).delete()


Item = namedtuple("Item", ['index', 'requiredTime', 'deadline'])
Product = namedtuple('Product', ['index', 'name', 'setupTime', 'unitProductionTime'])
Order_new = namedtuple('Order_new',
                       ['orderID', 'productIndex', 'quantity', 'productcode', 'deadline', 'workIndex', 'quality'])
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
                                int(lines[3]), int(lines[4]), int(lines[5])))
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
    ref_date_str = datetime.datetime.today().strftime('%m/%d/%Y %H:%M')
    ref_date = datetime.datetime.strptime(ref_date_str, "%m/%d/%Y %H:%M")
    m, s = divmod((date - ref_date).seconds, 60)
    return m * 100


def get_abs_timestamp(date):
    import time
    return time.mktime(date.timetuple())


def getstarttimestamp():
    import time
    import datetime
    ref_date_str = datetime.datetime.today().strftime('%m/%d/%Y %H:%M')
    return int(time.mktime(datetime.datetime.strptime(ref_date_str, "%m/%d/%Y %H:%M").timetuple()))


def timestamptodatestring(timestamp):
    global working_hrs
    dt_object = datetime.datetime.fromtimestamp(
        int(getstarttimestamp() + timestamp))
    dt_time = dt_object.time()
    if dt_time < working_hrs[0]:
        dt_object.replace(hour=working_hrs[0].hour, minute=working_hrs[0].minute)
    elif working_hrs[0] < dt_time < working_hrs[1]:
        dt_object = dt_object
    elif working_hrs[1] < dt_time < working_hrs[2]:
        dt_object.replace(hour=working_hrs[2].hour, minute=working_hrs[2].minute)
    elif working_hrs[2] < dt_time < working_hrs[3]:
        dt_object = dt_object
    elif dt_time > working_hrs[3]:
        dt_object += datetime.timedelta(days=1)
        dt_object.replace(hour=working_hrs[0].hour, minute=working_hrs[0].minute)
    return dt_object.strftime('%Y-%m-%d %H:%M')


def timestatmptodate(timestamp):
    import datetime
    return datetime.datetime.fromtimestamp(
        int(timestamp)
    )


def adjust_time(start_time_stamp, end_time_stamp):
    global working_hrs
    dt_object_start = timestatmptodate(start_time_stamp)
    dt_time_start = dt_object_start.time()

    dt_object_end = timestatmptodate(end_time_stamp)

    start_end_diff = dt_object_end - dt_object_start

    if dt_time_start < working_hrs[0]:
        dt_object_start = dt_object_start.replace(hour=working_hrs[0].hour, minute=working_hrs[0].minute)
        work_hr = int(working_hrs[0].hour) * 3600 + int(working_hrs[0].minute) * 60 + start_end_diff.seconds
        m, s = divmod(work_hr, 60)
        hr, m = divmod(m, 60)
        # if m>59:
        #     hr+=1
        # if hr<24:
        #     dt_object_end = dt_object_end.replace(hour=hr if hr<24 else 23,
        #                                       minute=m if m<60 else 0)
        # else:

    elif working_hrs[1] < dt_time_start < working_hrs[2]:
        dt_object_start = dt_object_start.replace(hour=working_hrs[2].hour, minute=working_hrs[2].minute)
        work_hr = int(working_hrs[2].hour) * 3600 + int(working_hrs[2].minute) * 60 + start_end_diff.seconds
        m, s = divmod(work_hr, 60)
        hr, m = divmod(m, 60)
        dt_object_end = dt_object_end.replace(hour=hr if hr<24 else 23,
                                              minute=m if m<60 else 0)
    elif dt_time_start > working_hrs[3]:
        dt_object_start += datetime.timedelta(days=1)
        dt_object_end += datetime.timedelta(days=1)
        work_hr = int(working_hrs[0].hour) * 3600 + int(working_hrs[0].minute) * 60 + start_end_diff.seconds
        m, s = divmod(work_hr, 60)
        hr, m = divmod(m, 60)
        dt_object_start = dt_object_start.replace(hour=working_hrs[0].hour, minute=working_hrs[0].minute)
        dt_object_end = dt_object_end.replace(hour=hr if hr<24 else 23,
                                              minute=m if m<60 else 0)
    dt_time_end_new = dt_object_end.time()
    dt_time_start_new = dt_object_start.time()
    start_end_diff = dt_object_end - dt_object_start

    if dt_time_start_new < working_hrs[1] and dt_time_end_new > working_hrs[1]:
        dt_object_start = dt_object_start.replace(hour=working_hrs[2].hour, minute=working_hrs[2].minute)
        work_hr = int(working_hrs[2].hour) * 3600 + int(working_hrs[2].minute) * 60 + start_end_diff.seconds
        m, s = divmod(work_hr, 60)
        h, m = divmod(m, 60)
        dt_object_end = dt_object_end.replace(hour=h if h<24 else 23,
                                              minute=m if m<60 else 0)
    elif dt_time_end_new > working_hrs[3]:
        dt_object_start += datetime.timedelta(days=1)
        dt_object_end += datetime.timedelta(days=1)
        dt_object_start = dt_object_start.replace(hour=working_hrs[0].hour, minute=working_hrs[0].minute)
        work_hr = int(working_hrs[0].hour) * 3600 + int(working_hrs[0].minute) * 60 + start_end_diff.seconds
        m, s = divmod(work_hr, 60)
        h, m = divmod(m, 60)
        dt_object_end = dt_object_end.replace(hour=h if h<24 else 23,
                                              minute=m if m<60 else 0)

    return dt_object_start.strftime('%Y-%m-%d %H:%M'), dt_object_end.strftime('%Y-%m-%d %H:%M'), get_abs_timestamp(
        dt_object_end)





# prod_time={'02':3.33/10,'05':2.86/10,'21':2.33/10}
quality_map={'0':['4','6','8'],'1':['7','9','11','12']}

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
    global completed_orders
    global quality_map
    prod_times=read_actual_productiontime()
    for index, row in order_data_df.iterrows():
        import re

        prod_type = re.findall("\d+", row['code of product'])[0]
        if len(prod_type)==4:
            quality_code=prod_type[:1]
        else:
            quality_code=prod_type[:2]

        if quality_code in quality_map['0']:
            quality=0
        else:
            quality=1
        prod_type = re.findall("\d+",  row['code of product'])[0][-2:] + str(quality)

        unit_time = prod_times[prod_type] if prod_type in prod_times else 3/30000
        total_time=row['Quantity']*unit_time
        max_order_quantity=int(4/unit_time)
        if str(row['Production Order Nr.']) not in completed_orders:
            orders_map.update(
                {row['Production Order Nr.']: [row['Name of product'], row['Quantity'], row['code of product']]})
            orders.append(
                Order_new(row['Production Order Nr.'], 1, row['Quantity'], row['code of product'],
                          int(gettimestamp(row['End Time'])),
                          index, quality))
        # if row['Quantity']>max_order_quantity:
        #     lot = max_order_quantity
        #     remaining=row['Quantity']
        #     count=1
        #     while True:
        #         if str(str(row['Production Order Nr.'])+'_'+str(count)) not in completed_orders:
        #             orders_map.update(
        #                 {str(row['Production Order Nr.'])+'_'+str(count): [row['Name of product'], lot, row['code of product']]})
        #             orders.append(
        #                 Order_new(str(row['Production Order Nr.'])+'_'+str(count), 1, lot, row['code of product'],
        #                           int(gettimestamp(row['End Time'])),
        #                           index, quality))
        #             count+=1
        #             remaining-=lot
        #             if remaining<=0:
        #                 import ipdb
        #                 ipdb.set_trace()
        #                 break
        #             if remaining>max_order_quantity:
        #                 lot=max_order_quantity
        #             else:
        #                 lot=remaining
        #
        #
        # else:



    # for i in range(1, order_count):
    #     lines = order_lines[i].split(',')
    #     orders.append(Order(int(lines[0]), int(lines[1]), int(lines[2]),
    #                         int(lines[3]), int(lines[4])))
    processed_orders = []
    for order in orders:
        # in format Order(requiredTime, deadline)
        product = [p for p in products if p.index == order.productIndex]
        import re

        prod_type = re.findall("\d+", order.productcode)[0][-2:] + str(order.quality)
        global prod_time
        unit_time = prod_time[prod_type] if prod_type in prod_time else 3/30000
        processed_orders.append([order.orderID, int(order.quantity * unit_time
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


def generateschduele(file_location="data/PolishingOrders_new.xlsx"):
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
    left = getstarttimestamp()
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
            value = order[1] * 10
            left += setupTime
            color_index = order[3]
            # x_ticks.append(left)
            if MaintenanceDate_start <= left + value and left <= MaintenanceDate_end:
                left = MaintenanceDate_end
            left += value
        start_dt, end_dt, left = adjust_time(start + setupTime, left)
        p_order = {'order_id': order_index, 'product_name': orders_map[order_index][0],
                   'quantity': orders_map[order_index][1], 'product_id': orders_map[order_index][2],
                   'start_datetime': start_dt,
                   'end_datetime': end_dt, 'status': 0}
        final_orders.append(p_order)
    # generate_schedule_graph(final_orders)
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
    DF = ps.DataFrame(columns=['Item', 'Start', 'End', 'Status', 'Color'])
    items = []
    for order in final_orders:
        l = [str(order['order_id']) + '-' + order['product_name'], order['start_datetime'], order['end_datetime']]
        if int(order['status']) == 0:
            l.append('Pending')
            l.append('Orange')
        elif int(order['status']) == 1:
            l.append('Progress')
            l.append('Blue')
        else:
            l.append('Completed')
            l.append('Green')
        items.append(l)
    for i, Dat in enumerate(items[::-1]):
        DF.loc[i] = Dat
    DF['Start_dt'] = ps.to_datetime(DF.Start)
    DF['End_dt'] = ps.to_datetime(DF.End)
    G = figure(title='Polishing Schedule', x_axis_type='datetime', width=1200, height=400, y_range=DF.Item.tolist(),
               x_range=Range1d(DF.Start_dt.min(), DF.End_dt.max(), min_interval=datetime.timedelta(minutes=30)))
    G.xaxis.formatter = DatetimeTickFormatter(
        hours=["%d %b %y, %H:%m"],
        days=["%d %b %y, %H:%m"],
        months=["%d %b %y, %H:%m"],
        years=["%d %b %y, %H:%m"],
    )
    G.xaxis.major_label_orientation = pi / 3
    tick_vals = pd.to_datetime(date_range(DF.Start_dt.min(), DF.End_dt.max(), 2, 'hours')).astype(int) / 10 ** 6
    G.xaxis.ticker = FixedTicker(ticks=list(tick_vals))
    hover = HoverTool(tooltips="Product: @Item<br>\
    Start: @Start<br>\
    End: @End<br>\
    Status: @Status")
    G.add_tools(hover)

    DF['ID'] = DF.index + 0.8
    DF['ID1'] = DF.index + 1.2
    CDS = ColumnDataSource(DF)
    G.quad(left='Start_dt', right='End_dt', bottom='ID', top='ID1', source=CDS, color="Color", legend='Status',
           alpha=0.8
           )
    G.legend.click_policy = "hide"
    # G.rect(,"Item",source=CDS)
    # show(G)
    export_png(G, filename="static/schedule.png")
    output_file('static/schedule.html', mode='inline')
    save(G)


def maintainprocessedhistory():
    print('history maintenance')
