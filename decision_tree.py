import pandas as pd 
df = pd.read_csv('/home/shaik/Downloads/archive (6)/BankCustomerData.csv')
print(df)
print(df.info())
df = df['marital'].value_counts()
print(df)

import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('/home/shaik/Downloads/archive (6)/BankCustomerData.csv')

# Replace 'Column_Name' with the name of the column you want to find unique values for
unique_values = df['poutcome'].unique()

# Print the unique values
print(unique_values)

import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('/home/shaik/Downloads/archive (6)/BankCustomerData.csv')

# Define the mapping for replacement
convert = {"marital": {"married": 0, "single": 1,"divorced" : 2},
          "housing":{"yes":1,"no":0},
            "loan":{"no":0,"yes":1},
           "default":{"yes":1,"no":0},
           "poutcome":{"unknown":0,"success":1,"other":2,"failure":3},
           "month" : {
                "jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5,
                "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11
            },
           #"term_deposit":{"yes":1,"no":0}
           "job":{"management":0,"technician":1,"entrepreneur":2,"blue-collar":3,"unknown":4,"student":5,"retired":6,"services":7,"admin.":8,"unemployed":9,"self-employed":10,"housemaid":11},
          "education":{"primary":0,"secondary":1,"tertiary":2,"unknown":3},
           "contact":{"cellular":0,"telephone":1,"unknown":2}
          }
# Replace values in the 'marital' column using the defined mapping
df.replace(convert, inplace=True)
# Print the modified DataFrame
print(df)


xc = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']
y=['yes','no']
all_inputs = df[xc]
all_classes=df['term_deposit']

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

(x_train,x_test,y_train,y_test)=train_test_split(all_inputs,all_classes,train_size=0.7)

clf = DecisionTreeClassifier(random_state=0)

clf.fit(x_train,y_train)
score = clf.score(x_test,y_test)
print(score)


from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
import graphviz


dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True ,rounded = True,feature_names = xc,class_names=y)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.set_size('"2000,2000"')
graph.write_png('bankTreeTrain.png')
Image(graph.create_png())