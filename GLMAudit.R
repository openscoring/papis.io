library("r2pmml")

audit = read.csv("csv/Audit.csv")
audit$Adjusted = as.factor(audit$Adjusted)

ageQuantiles = quantile(audit$Age)

glm = glm(Adjusted ~ . - Age + cut(Age, breaks = ageQuantiles) + Gender:Education + I(Income / (Hours * 52)), data = audit, family = "binomial")

audit$Adjusted = NULL

glm = r2pmml::verify(glm, audit[sample(nrow(audit), 100), ])

r2pmml::r2pmml(glm, "pmml/GLMAudit.pmml")
