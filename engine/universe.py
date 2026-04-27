class Universe:
    _tick = 0
    id = 0
    tick_to_second = 0.5
    landscape = None
    graph = None
    graph_pod = None
    deadlock_prevention_manager = None

    def __init__(self):
        self._objects = []          # instance-level, not shared across instances
        self.warehouse_size = []    # instance-level

    def addObject(self, object):
        object.id = len(self._objects)
        object.setUniverse(self)
        self._objects.append(object)
    
    def set_warehouse_size(self, size):
        self.warehouse_size = size

    def get_warehouse_size(self):
        return self.warehouse_size

    def tick(self):
        for o in self._objects:
            o.move()

    def get_movable_objects(self):
        return self._objects
    
    def generateResult(self):
        # print("\n\ngenerateResult is called\n\n")
        # print("moveable objects")
        # print(len(self.get_movable_objects()))
        result = []
        for o in self.get_movable_objects():
            result.append({
                'id': o.id,
                'heading': o.heading,
                'shape': o.shape,
                'velocity': o.velocity,
                'acceleration': o.acceleration,
                'pos_x': o.pos_x,
                'pos_y': o.pos_y,
                'color': o.color,
            })

        return result

