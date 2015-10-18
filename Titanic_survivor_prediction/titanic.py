import pandas as pd
from collections import Counter
c = pd.read_csv("./train.csv")
c["newSex"] = c["Sex"].map({"female":0,"male":1}).astype(int)
print c[0:10][["Sex","newSex"]]
# print c['Age'][10:20]
# s = c[(c.Survived == 1)]
# print len(s[s.Fare>10]),len(s)
# t = pd.read_csv("./test.csv")
# Survived = [0 for i in range(len(t))]
# for j in range(len(t)):
# 	age = t["Age"][j]
# 	if  age<18 or age>60 or t.iloc[j][t.columns.get_loc("Sex")]=="female":
# 		Survived[j]=1
# t["Survived"] = Survived
# t.to_csv("newtest.csv",cols=["PassengerId","Survived"])