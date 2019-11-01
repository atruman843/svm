from read_file import Read_File
from data_processor import Data_Processor
from svm import SVM
from decision_tree import Decision_Tree

fileName = 'data/census-income_10percentData.csv'

file_reader = Read_File(fileName)
file_reader.read()

features    = file_reader.get_features()
labels      = file_reader.get_labels()

data_fill   = Data_Processor(features)
data_fill.fill_empty_fields()

# will perform svm task 
my_svm = SVM(features, labels)
my_svm.calculate_info_gain()
my_svm.stratified_k_fold(10)
my_svm.svm()
my_svm.draw_svm()

my_tree = Decision_Tree(features, labels)
my_tree.calculate_info_gain()
for i in range(1, 14):
    print "============================{}============================".format(i)
    my_tree.decision(i)
