from typing import List, Dict, Optional

from model.order import Order


class OrderManager:
    def __init__(self):
        self.orders: List[Order] = []
        self.order_id_to_order: Dict[int, Order] = {}
        # self.finished_orders: List[Order] = []
        self.unfinished_orders: List[Order] = []
        self.preassign_order_ids: List[int] = []

    def add_order(self, order: Order):
        self.orders.append(order)
        self.order_id_to_order[order.order_id] = order
        self.unfinished_orders.append(order)

    def get_order_by_id(self, order_id) -> Optional[Order]:
        """Retrieve an order by its ID using the dictionary for quick access."""
        return self.order_id_to_order.get(order_id, None)

    def remove_order(self, order:Order):
        self.orders.remove(order)

    def finish_order(self, order_id, tick: int):
        """Move an order from the unfinished_orders list to the finished_orders list."""
        order = self.get_order_by_id(order_id)
        order.complete_order(tick)
        if order and order in self.unfinished_orders:
            self.unfinished_orders.remove(order)
            # self.finished_orders.append(order)
