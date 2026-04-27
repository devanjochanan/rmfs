from datetime import datetime

class Landscape:
    dimension = 0
    total_objects = 0
    _map = []
    _objects = {}

    def __init__(self, dimension):
        self.dimension = dimension
        self.total_objects = 0
        self._objects = {}
        self._map = []  # instance-level, not shared across instances
        self.current_date_string = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        for i in range(self.dimension+1):
            one_row = []
            for j in range(self.dimension+1):
                one_row.append([])
            self._map.append(one_row)
    
    def get_robot_object(self):
        return self._objects
    
    def _clamp(self, x, y):
        """Clamp coordinates to valid grid bounds [0, dimension]."""
        cx = max(0, min(self.dimension, round(x)))
        cy = max(0, min(self.dimension, round(y)))
        return cx, cy

    def _setObjectNew(self, label, x, y, speed, acceleration, heading, state, load_mass):
        self.total_objects += 1

        movement = 'vertical'
        if heading == 270 or heading == 90:
            movement = 'horizontal'

        self._objects[label] = {
            'label': label,
            'x': x,
            'y': y,
            'velocity': speed,
            'acceleration': acceleration,
            'heading': heading,
            'movement': movement,
            'state': state,
            'load_mass': load_mass,
        }

        cx, cy = self._clamp(x, y)
        self._map[cx][cy].append(self._objects[label])

    def setObject(self, label, x, y, speed, acceleration, heading, state, load_mass):
        if label not in self._objects:
            return self._setObjectNew(label, x, y, speed, acceleration, heading, state, load_mass)

        old_cx, old_cy = self._clamp(self._objects[label]['x'], self._objects[label]['y'])
        new_cx, new_cy = self._clamp(x, y)

        # check if x or y has changed
        if new_cx != old_cx or new_cy != old_cy:
            # remove from old position
            to_iter = self._map[old_cx][old_cy]
            for index, e in enumerate(to_iter):
                if e['label'] == label:
                    del to_iter[index]
                    break

            # add to new position
            self._map[new_cx][new_cy].append(self._objects[label])

        movement = 'vertical'
        if heading == 270 or heading == 90:
            movement = 'horizontal'

        self._objects[label] = {
            'label': label,
            'x': x,
            'y': y,
            'velocity': speed,
            'acceleration': acceleration,
            'heading': heading,
            'movement': movement,
            'state': state,
            'load_mass': load_mass,
        }

    def getNeighborObject(self, x, y, radius):
        i = x-radius
        j = y+radius
        check = 2*radius+1
        points_to_check = []
        result = []
        while i < x+check:
            j = y+radius
            while j > y-check:
                if 0 <= i <= self.dimension and 0 <= j <= self.dimension:
                    if i != x or j != y:
                        points_to_check.append([i, j])
                j -= 1
            i += 1

        for p in points_to_check:
            s = self._map[p[0]][p[1]]
            if len(s) > 0:
                for obj in s:
                    result.append(self._objects[obj['label']])

        return result

    def get_neighbor_object(self, x, y):
        cx, cy = self._clamp(x, y)
        s = self._map[cx][cy]
        if len(s) > 0:
            for obj in s:
                return self._objects[obj['label']]
        return None

    @property
    def objects(self):
        return self._objects

        