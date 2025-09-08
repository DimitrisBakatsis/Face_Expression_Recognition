## R script for Topic 5 of DAMA51 HW5
#1
cat('\nQuestion 1')
cat('\n-----------\n')
cat('\nOpen and read the dataset and remove any unnecessary columns\n\n')
file_path<-file.choose()
library("readxl")
face_expression<-read_xls(file_path)
face_expression <- face_expression[ ,-c(26,27,28,30,31,32,33,34,35,36)]
View(face_expression)
cat('\nStandardize the data and use the PCA technique to extract the principal components\n')
face_expression_scaled <- scale(face_expression[, 1:25])
pca_result <- prcomp(face_expression_scaled)
pca_variance<-summary(pca_result)$importance[2,]
cat('\nCreation of a Scree plot to see visually the most important Principal Components\n')
plot(pca_variance, type = "b", pch = 19, col = "blue", 
     xlab = "Principal Components", ylab = "Proportion of Variance",
     main = "Scree Plot")
text(x = 1:length(pca_variance), 
     y = pca_variance, 
     labels = paste("PC", 1:length(pca_variance), sep = ""), 
     pos = 3, cex = 0.8, col = "red")
cat('\nCalculation of the most important Principal Components by setting a threshold of importance of variance\n')
num_components <- which(cumsum(pca_variance) >= 0.90)[1]
cat('\nCalculation of the original values of each PCA for every measurement and their absolute values\n')
selected_pcs <- pca_result$x[, 1:num_components]
loadings <- pca_result$rotation[, 1:num_components]
abs_loadings <- abs(loadings)
cat('\nSum of each row horizontally to find the total value of each measurement\n')
feature_importance <- rowSums(abs_loadings[, 1:num_components])
cat('\nSort of the measurements from the most fundamental to the least one\n')
names(feature_importance) <- rownames(abs_loadings)
sorted_features_pca <- sort(feature_importance, decreasing = TRUE)
sorted_features_pca <- names(sorted_features_pca)
print(sorted_features_pca)

# Function to perform Chi-Square test for each feature with respect to 'Expression'
chi_squared_results <- function(data, target) {
  p_values <- sapply(data, function(feature) {
    # Convert continuous features to categorical (if necessary)
    feature_cat <- cut(feature, breaks = 7)  # Binning continuous data into 5 categories
    
    # Perform Chi-Square test
    chisq_test <- chisq.test(table(feature_cat, target))
    return(chisq_test$p.value)  # Extract p-value
  })
  return(p_values)
}

# Run Chi-Square test for each feature (1st to 25th columns) with respect to the 'Expression' target
chi_sq_p_values <- chi_squared_results(face_expression[,1:25],face_expression$Expression)

# Sort the features by their p-values (from smallest to largest)
sorted_chi_sq_p_values <- sort(chi_sq_p_values,decreasing=TRUE)

# Print sorted p-values to see which features are most significant
cat("\nSorted Features by Chi-Square p-values:\n")
print(sorted_chi_sq_p_values)

# Get the names of the sorted features
sorted_features_chi_sq <- names(sorted_chi_sq_p_values)
cat("\nSorted Feature Names by Chi-Square p-value:\n")
print(sorted_features_chi_sq)

# Convert 'Expression' to numeric (if it is a factor or character variable)
face_expression$Expression_numeric <- as.numeric(factor(face_expression$Expression))

# Combine the scaled features and the numeric 'Expression' column into a new data frame
data_with_target <- cbind(face_expression_scaled, Expression = face_expression$Expression_numeric)

# Compute the correlation matrix between all features and the target 'Expression'
cor_matrix <- cor(data_with_target)

# Extract the correlation of each feature with the 'Expression' variable
feature_cor_with_target <- cor_matrix[, "Expression"]
heatmap(cor_matrix)
# Sort features by their absolute correlation with 'Expression'
cat("Most fundamental features based on correlation with Expression:\n")
sorted_features_corr <- sort(abs(feature_cor_with_target), decreasing = TRUE)
sorted_features_corr<-names(sorted_features_corr)
sorted_features_corr <- setdiff(sorted_features_corr, "Expression")
print(sorted_features_corr)

library(randomForest)
# Train a Random Forest model (using the combined dataset)
rf_model <- randomForest(Expression ~ ., data = data_with_target)
# Get feature importance from the Random Forest model
feature_importance_rf <- rf_model$importance
sorted_features_rf <- names(sort(feature_importance_rf[, 1], decreasing = TRUE))

# Print sorted features
print(sorted_features_rf)

# Find the top five common features of the above four methods
common_features <- intersect(intersect(intersect(sorted_features_pca, sorted_features_chi_sq),sorted_features_corr),sorted_features_rf)
cat("Most fundamental features based on combination of four different methods:\n")
head(common_features,5)

#2
install.packages("caret")
library(caret)

# Set seed for reproducibility
set.seed(123)

# Split the dataset into training (60%), validation (20%), and test (20%) sets
train_index <- createDataPartition(face_expression$Expression, p = 0.6, list = FALSE)
train_data <- face_expression[train_index, ]
temp_data <- face_expression[-train_index, ]
valid_index <- createDataPartition(temp_data$Expression, p = 0.5, list = FALSE)
validation_data <- temp_data[valid_index, ]
test_data <- temp_data[-valid_index, ]
train_data$Expression <- factor(train_data$Expression)
validation_data$Expression <- factor(validation_data$Expression)
test_data$Expression <- factor(test_data$Expression)

# Train a model using Random Forest
rf_model <- randomForest(Expression ~ ., data = train_data)

# Predict on test set
test_pred <- predict(rf_model, newdata = test_data)

# Confusion Matrix for the test set
conf_matrix <- confusionMatrix(factor(test_pred), factor(test_data$Expression))
print(conf_matrix)

# Calculate accuracy
total <- sum(conf_matrix$table)  # Total number of observations
accuracy <- round(sum(diag(conf_matrix$table)) / total,3)
cat("Accuracy:", accuracy, "\n")

# Calculate Precision and Recall for each class
precision <- round(conf_matrix$byClass[, "Precision"],3)
recall <- round(conf_matrix$byClass[, "Recall"],3)
f1_score <- round(2 * (precision * recall) / (precision + recall),3)

# Print results
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")

# Calculate macro-average Precision, Recall, and F1-Score
avg_precision <- round(mean(precision, na.rm = TRUE),3)
avg_recall <- round(mean(recall, na.rm = TRUE),3)
avg_f1_score <- round(mean(f1_score, na.rm = TRUE),3)

# Print results
cat("Average Precision:", avg_precision, "\n")
cat("Average Recall:", avg_recall, "\n")
cat("Average F1 Score:", avg_f1_score, "\n")

#3
#Random Forest model Classifier
rf_model <- randomForest(Expression ~ ., data = train_data)

# View the model summary
print(rf_model)
# Make predictions on the test data
rf_pred <- predict(rf_model, newdata = test_data)

# View the predictions
print(rf_pred)

# Confusion Matrix
conf_matrix_rf <- confusionMatrix(factor(rf_pred), factor(test_data$Expression))
print(conf_matrix_rf)

# Calculate Precision and Recall for each class
precision_rf <- round(conf_matrix_rf$byClass[, "Precision"],3)
recall_rf <- round(conf_matrix_rf$byClass[, "Recall"],3)
f1_score_rf <- round(2 * (precision_rf * recall_rf) / (precision_rf + recall_rf),3)

# Calculate macro-average Precision, Recall, and F1-Score
avg_precision_rf <- round(mean(precision_rf, na.rm = TRUE),3)
avg_recall_rf <- round(mean(recall_rf, na.rm = TRUE),3)
avg_f1_score_rf <- round(mean(f1_score_rf, na.rm = TRUE),3)

# Print results
cat("Average Precision:", avg_precision_rf, "\n")
cat("Average Recall:", avg_recall_rf, "\n")
cat("Average F1 Score:", avg_f1_score_rf, "\n")


#Naive Bayes Classifier
install.packages("e1071")
library(e1071)
nb_model <- naiveBayes(Expression ~ ., data = train_data)
print(nb_model)
# Make predictions on the test data
nb_pred <- predict(nb_model, newdata = test_data)

# View the predictions
print(nb_pred)
# Confusion Matrix
conf_matrix_nb <- confusionMatrix(factor(nb_pred), factor(test_data$Expression))
print(conf_matrix_nb)

# Calculate Precision and Recall for each class
precision_nb <- round(conf_matrix_nb$byClass[, "Precision"],3)
recall_nb <- round(conf_matrix_nb$byClass[, "Recall"],3)
f1_score_nb <- round(2 * (precision_nb * recall_nb) / (precision_nb + recall_nb),3)

# Calculate macro-average Precision, Recall, and F1-Score
avg_precision_nb <- round(mean(precision_nb, na.rm = TRUE),3)
avg_recall_nb <- round(mean(recall_nb, na.rm = TRUE),3)
avg_f1_score_nb <- round(mean(f1_score_nb, na.rm = TRUE),3)

# Print results
cat("Average Precision:", avg_precision_nb, "\n")
cat("Average Recall:", avg_recall_nb, "\n")
cat("Average F1 Score:", avg_f1_score_nb, "\n")


# Apply KNN Classifier
library(class)
k_value <- 5  # Choose the number of neighbors (k)
knn_pred <- knn(train = train_data[, -which(names(train_data) == "Expression")],
                test = test_data[, -which(names(test_data) == "Expression")],
                cl = train_data$Expression, k = k_value)

# Generate the confusion matrix for KNN predictions
knn_conf_matrix <- confusionMatrix(factor(knn_pred), factor(test_data$Expression))
print(knn_conf_matrix)

# Calculate evaluation metrics (accuracy, precision, recall, F1 score) for each classifier
rf_accuracy <- round(sum(diag(conf_matrix_rf$table)) / sum(conf_matrix_rf$table),3)
nb_accuracy <- round(sum(diag(conf_matrix_nb$table)) / sum(conf_matrix_nb$table),3)
knn_accuracy <- round(sum(diag(knn_conf_matrix$table)) / sum(knn_conf_matrix$table),3)

# Calculate Precision and Recall for each class
precision_knn <- round(knn_conf_matrix$byClass[, "Precision"],3)
recall_knn <- round(knn_conf_matrix$byClass[, "Recall"],3)
f1_score_knn <- round(2 * (precision_knn * recall_knn) / (precision_knn + recall_knn),3)

# Calculate macro-average Precision, Recall, and F1-Score
avg_precision_knn <- round(mean(precision_knn, na.rm = TRUE),3)
avg_recall_knn <- round(mean(recall_knn, na.rm = TRUE),3)
avg_f1_score_knn <- round(mean(f1_score_knn, na.rm = TRUE),3)

# Print the results
cat("Random Forest Accuracy:", rf_accuracy, "\n")
cat("Naive Bayes Accuracy:", nb_accuracy, "\n")
cat("KNN Accuracy:", knn_accuracy, "\n")
cat("Random Forest Average Precision:", avg_precision_rf, "\n")
cat("Naive Bayes Average Precision:", avg_precision_nb, "\n")
cat("KNN Average Precision:", avg_precision_knn, "\n")
cat("Random Forest Average Recall:", avg_recall_rf, "\n")
cat("Naive Bayes Average Recall:", avg_recall_nb, "\n")
cat("KNN Average Recall:", avg_recall_knn, "\n")
cat("Random Forest Average F1 Score:", avg_f1_score_rf, "\n")
cat("Naive Bayes Average F1 Score:", avg_f1_score_nb, "\n")
cat("KNN Average F1 Score:", avg_f1_score_knn, "\n")

#4
true_labels <- face_expression$Expression
face_expression_scaled <- scale(face_expression[, 1:25])

#Apply K-means clustering method
cat("\nApplying K-Means Clustering method\n")
set.seed(123)
kmeans_result <- kmeans(face_expression_scaled, centers = length(unique(true_labels)), nstart = 25)
kmeans_clusters <- kmeans_result$cluster
summary(kmeans_clusters)

#Apply Hierarchical clustering method
cat("\nApplying Hierarchical Clustering method\n")
dist_matrix <- dist(face_expression_scaled)
hc_result <- hclust(dist_matrix)
hc_clusters <- cutree(hclust(dist_matrix), k = length(unique(true_labels)))
summary(hc_clusters)

#Apply DBSCAN clustering method
install.packages("dbscan")
library(dbscan)
cat("\nApplying DBSCAN Clustering method\n")
dbscan_result <- dbscan(face_expression_scaled, eps = 2, minPts = 5)
dbscan_clusters <- dbscan_result$cluster
summary(dbscan_clusters)

#Evaluation of the clustering methods using plots
# First two principal components
pca_2d <- prcomp(face_expression_scaled)$x[, 1:2]

# Plot K-means clusters
plot(pca_2d, col = kmeans_clusters, main = "K-Means Clusters", pch = 19)

# Plot Hierarchical clustering
plot(pca_2d, col = hc_clusters, main = "Hierarchical Clustering", pch = 19)

# Plot DBSCAN clustering
dbscan_clusters_color <- ifelse(dbscan_clusters == 0, "gray", rainbow(length(unique(dbscan_clusters)))[dbscan_clusters])
plot(pca_2d, col = dbscan_clusters_color, main = "DBSCAN Clusters", pch = 19)

# Plot true labels
label_numeric <- as.numeric(factor(true_labels))
plot(pca_2d, col = label_numeric, main = "True Labels", pch = 19)
