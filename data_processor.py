import statistics

class Data_Processor:

    def __init__(self, features):
        self.features = features
        self.fill     = []
        self.sets     = []
        self.counter  = 0

    def fill_empty_fields(self):
        for ind, field in enumerate(self.features):
            if type(field[0]) is str:
                self.fill.append(statistics.mode(field))
                self.create_unique_list(ind)
            else:
                self.fill.append(statistics.mean(field))

        print self.fill

        for ind1, field in enumerate(self.features):
            for ind2, value in enumerate(field):
                if value == '?':
                    self.features[ind1][ind2] = self.fill[ind1]
                if type(self.features[ind1][ind2]) is str:
                    self.clean_categorical(ind1, ind2)

    def clean_categorical(self, ind1, ind2):
        for set in self.sets:
            if self.features[ind1][ind2] in set:
                self.features[ind1][ind2] = set.index(self.features[ind1][ind2])
                return 0

    def create_unique_list(self, ind):
        self.sets.append([])
        for item in self.features[ind]:
            if not(item in self.sets[self.counter]):
                self.sets[self.counter].append(item)
        self.counter += 1
