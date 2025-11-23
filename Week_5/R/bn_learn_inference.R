# bn_learn_inference.R
# Run in R or RStudio
# Required packages: bnlearn, gRain, bnclassify
# install.packages(c("bnlearn","gRain","bnclassify"))

library(bnlearn)
library(gRain)

# ---------------------------
# Load data
# ---------------------------
# Example: grades.csv should have columns: EC100, IT101, MA101, PH100, Qualify
# All columns are categorical (factors)
data_path <- "data/student_grades.csv"
grades <- read.csv(data_path, stringsAsFactors = TRUE)

# quick look
str(grades)
head(grades)

# ---------------------------
# Structure learning (Hill-Climbing)
# ---------------------------
# We use score = "bic" or "aic" or "bde"
hc_net <- hc(grades, score = "bic")
plot(hc_net)   # save a figure to figures/fig_bn_structure.png if desired

# Save the structure plot
png("figures/fig_bn_structure.png", width=900, height=600)
graphviz.plot(hc_net)
dev.off()

# ---------------------------
# Parameter learning (CPTs)
# ---------------------------
fitted <- bn.fit(hc_net, data = grades, method = "mle")

# Example: show CPT for EC100
print(fitted$EC100)

# To visualize a particular CPT: we can plot barplots and save
png("figures/fig_cpt_ec100.png", width=600, height=400)
barplot(fitted$EC100$prob, main = "CPT: EC100 (marginal)", ylab = "Probability")
dev.off()

# ---------------------------
# Inference with gRain
# ---------------------------
grain_net <- as.grain(fitted)
# compile
g <- compile(grain_net)

# Example query:
# Given EC100 = "DD", IT101="CC", MA101="CD" -> most probable PH100?
ev <- setEvidence(g, evidence = list(EC100="DD", IT101="CC", MA101="CD"))
query <- querygrain(ev, nodes = c("PH100"), type = "marginal")
print(query)

# ---------------------------
# Save fitted model for reuse
# ---------------------------
saveRDS(fitted, file = "R/fitted_bn.rds")
