# nb_tan_classifiers.R
# Naive Bayes and TAN using bnclassify
# install.packages("bnclassify")

library(bnclassify)
library(bnlearn)
library(caret)   # for cross-validation

# Load data
grades <- read.csv("data/student_grades.csv", stringsAsFactors = TRUE)

# make target variable 'Qualify' as factor if not
grades$Qualify <- as.factor(grades$Qualify)

# Split into train/test multiple times and report accuracy
set.seed(123)
reps <- 20
acc_nb <- numeric(reps)
acc_tan <- numeric(reps)

for (i in 1:reps) {
  idx <- createDataPartition(grades$Qualify, p = 0.7, list=FALSE)
  train <- grades[idx,]
  test  <- grades[-idx,]

  # Naive Bayes
  nb_model <- NB(train, class = "Qualify")
  pred_nb <- predict(nb_model, test)
  acc_nb[i] <- mean(pred_nb == test$Qualify)

  # TAN: learn structure using tan_cl is built-in
  tan_model <- TAN(train, class = "Qualify")
  pred_tan <- predict(tan_model, test)
  acc_tan[i] <- mean(pred_tan == test$Qualify)
}

cat("Naive Bayes mean accuracy:", mean(acc_nb), "\n")
cat("TAN mean accuracy:", mean(acc_tan), "\n")

# Save a small plot
png("figures/fig_nb_accuracy.png", width=700, height=400)
boxplot(acc_nb, acc_tan, names=c("NB","TAN"), main="Accuracy over 20 runs")
dev.off()
