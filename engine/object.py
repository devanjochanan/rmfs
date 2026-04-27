from .universe import Universe


class Object:
    def __init__(self):
        self.universe = None
        self.heading = 0
        self.velocity = 0
        self.acceleration = 0
        self.pos_x = 0
        self.pos_y = 0
        self.color = 15  # red

    def setUniverse(self, universe: Universe):
        self.universe = universe

    def move(self):
        if self.velocity != 0 or self.acceleration != 0:
            if self.acceleration != 0:
                self.velocity = self.velocity + (self.acceleration * self.universe.tick_to_second)

            if self.heading == 0:
                self.pos_y += self.velocity * self.universe.tick_to_second
            elif self.heading == 180:
                self.pos_y -= self.velocity * self.universe.tick_to_second
            elif self.heading == 90:
                self.pos_x += self.velocity * self.universe.tick_to_second
            elif self.heading == 270:
                self.pos_x -= self.velocity * self.universe.tick_to_second

    def rotate(self):
        self.heading += 90
        if self.heading == 360:
            self.heading == 0

    def rotateCC(self):
        if self.heading == 0:
            self.heading = 270
            return

        self.heading -= 90
