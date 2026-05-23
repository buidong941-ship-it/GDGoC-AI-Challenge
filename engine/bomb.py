class Bomb:
    def __init__(self, x, y, owner_id, radius=1, timer=7):
        self.x = x
        self.y = y
        self.owner_id = owner_id
        self.radius = radius
        self.timer = timer
        self.exploded = False
    
    def step(self):
        if self.exploded:
            return False
        self.timer -= 1
        if self.timer <= 0:
            self.exploded = True
            return True
        return False