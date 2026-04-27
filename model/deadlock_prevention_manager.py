from engine.netlogo_coordinate import NetLogoCoordinate


class DeadlockPreventionManager:
    def __init__(self, dimension):
        self._map = []
        for i in range(dimension + 1):
            one_row = []
            for j in range(dimension + 1):
                one_row.append(0)
            self._map.append(one_row)

    def bookCoordinate(self, p: NetLogoCoordinate):
        self._map[int(p.x)][int(p.y)] += 1

    def releaseCoordinate(self, p: NetLogoCoordinate):
        self._map[int(p.x)][int(p.y)] -= 1

    def coordinateBooked(self, p: NetLogoCoordinate):
        return self._map[int(p.x)][int(p.y)] > 0
