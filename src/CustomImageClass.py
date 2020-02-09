class CustomImageClass:
    def __init__(self, minY, maxY, minX, maxX):
        self.minY = minY
        self.maxY = maxY
        self.minX = minX
        self.maxX = maxX

    def getMinY(self):
        return self.minY

    def getMaxY(self):
        return self.maxY

    def getMinX(self):
        return self.minX

    def getMaxX(self):
        return self.maxX
