import numpy
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, explained_variance_score
from info_gain import info_gain

class Decision_Tree:

    def __init__(self, features, labels):
        self.features = features
        self.labels   = labels

    def decision(self, numFeats):
        self.important_feats = []
        for i in range(0, numFeats):
            self.important_feats.append(self.features[i])
        self.important_feats = numpy.transpose(self.important_feats)
        self.labels          = numpy.array(self.labels)
        self.dec_tree        = tree.DecisionTreeClassifier()
        feat_train, feat_test, label_train, label_test = train_test_split(self.important_feats, self.labels, test_size=0.2)
        self.dec_tree.fit(feat_train, label_train)
        print "SCORES: {precision}, {recall}, {f1}, {variance}".format(precision=precision_score(self.labels, self.dec_tree.predict(self.important_feats)),
                                                                                                   recall=recall_score(self.labels, self.dec_tree.predict(self.important_feats)),
                                                                                                   f1=f1_score(self.labels, self.dec_tree.predict(self.important_feats)),
                                                                                                   variance=explained_variance_score(self.labels, self.dec_tree.predict(self.important_feats)))


    def calculate_info_gain(self):
        self.gain = []
        for index, feature in enumerate(self.features):
            self.gain.append([info_gain.info_gain(feature, self.labels), index])
        self.gain.sort(key=self.take_first, reverse=True)
        print "======================================================="
        print "{first} {second}".format(first=self.gain[0], second=self.gain[1])
        print "======================================================="

    def take_first(self, element):
        return element[0]
