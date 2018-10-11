import pandas

iris_df = pandas.read_csv("csv/Iris.csv")
print(iris_df.head(5))

iris_X = iris_df[iris_df.columns.difference(["Species"])]
iris_y = iris_df["Species"]
