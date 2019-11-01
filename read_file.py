import csv

class Read_File:

    def __init__(self, file):
        self.file = file
        self.labels = []
        self.features = []

    def get_features(self):
        return self.features

    def get_labels(self):
        return self.labels

    def read(self):

        with open(self.file) as csvfile:
            reader = csv.reader(csvfile)
            reader.next()
            for ind1, row in enumerate(reader):
                for ind2, value in enumerate(row):
                    if ind2 == len(row)-1:
                        if (value == '<=50K'):
                            self.labels.append(0)
                        else:
                            self.labels.append(1)
                    else:
                        if ind1 == 0:
                            self.features.append([])
                        if value.isdigit():
                            value = int(value)
                            if value > 1000:
                                value /= 100000
                        self.features[ind2].append(value)
