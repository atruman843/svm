import numpy
from matplotlib import pyplot
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, explained_variance_score
from info_gain import info_gain

class SVM:

    def __init__(self, features, labels):
        self.features = features
        self.labels   = labels
        self.k_fold   = None

    def stratified_k_fold(self, k):
        self.k = k
        self.k_fold = StratifiedKFold(n_splits=k)

    def svm(self):

        self.important_feats = numpy.array([self.features[self.gain[0][1]], self.features[self.gain[1][1]]])
        # self.important_feats = self.features
        self.important_feats = numpy.transpose(self.important_feats)
        self.labels          = numpy.array(self.labels)
        # c = 0.001
        # while (c<=7):
        self.svc             = svm.SVC(kernel='rbf')
        count                = 1
        # print "========================C = {}========================".format(c)
        for train, test in self.k_fold.split(self.important_feats, self.labels):
            self.svc.fit(self.important_feats[train], self.labels[train])
            print "{training}, {test}".format(training=accuracy_score(self.labels[train], self.svc.predict(self.important_feats[train])),
                                              test=accuracy_score(self.labels[test], self.svc.predict(self.important_feats[test])))
            # print "ITERATION {}".format(count)
            # count += 1
            # c += 3
        print "SCORES: {precision}, {recall}, {f1}, {variance}".format(precision=precision_score(self.labels[train], self.svc.predict(self.important_feats[train])),
                                                                                                   recall=recall_score(self.labels[train], self.svc.predict(self.important_feats[train])),
                                                                                                   f1=f1_score(self.labels[train], self.svc.predict(self.important_feats[train])),
                                                                                                   variance=explained_variance_score(self.labels[train], self.svc.predict(self.important_feats[train])))

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

    def draw_svm(self):

        # plot the data using the two most important features; change color based on the labels
        pyplot.scatter(self.important_feats[:, 0], self.important_feats[:, 1], c=self.labels, s=50, cmap='binary')

        #############################################################################################
        ####################### code from sklearn to draw support vectors ###########################
        #############################################################################################
        change_axis = pyplot.gca()
        x_lim = change_axis.get_xlim()
        y_lim = change_axis.get_ylim()
        xx = numpy.linspace(x_lim[0], x_lim[1], 30)
        yy = numpy.linspace(y_lim[0], y_lim[1], 30)
        YY, XX = numpy.meshgrid(yy, xx)
        xy = numpy.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.svc.decision_function(xy).reshape(XX.shape)

        change_axis.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5)

        change_axis.scatter(self.svc.support_vectors_[:, 0], self.svc.support_vectors_[:, 1], s=100,
                            linewidth=1, facecolors='none', edgecolors='b')
        #############################################################################################
        #############################################################################################

        pyplot.show()
