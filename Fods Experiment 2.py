import pandas as pd
data = {'customer_id': [1, 1, 2, 2, 3],
        'order_date': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-03', '2023-01-02'],
        'product_name': ['A', 'B', 'A', 'C', 'B'],
        'order_quantity': [2, 3, 1, 4, 2]}
order_data = pd.DataFrame(data)
total_orders_by_customer = order_data.groupby('customer_id').size()
print("Total Orders by Each Customer:\n", total_orders_by_customer)
average_quantity_per_product = order_data.groupby('product_name')['order_quantity'].mean()
print("\nAverage Order Quantity for Each Product:\n", average_quantity_per_product)
earliest_order_date = order_data['order_date'].min()
latest_order_date = order_data['order_date'].max()
print("\nEarliest Order Date:", earliest_order_date)
print("Latest Order Date:", latest_order_date)
