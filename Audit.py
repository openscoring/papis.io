import pandas

audit_df = pandas.read_csv("csv/Audit.csv")
print(audit_df.head(5))

audit_X = audit_df[audit_df.columns.difference(["Adjusted"])]
audit_y = audit_df["Adjusted"]
