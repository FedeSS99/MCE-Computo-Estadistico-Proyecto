library(dplyr)
library(ggplot2)
library(glmnet)
library(pls)

source("./CargaDatos.R")
source("./Rutinas.R")

# ------------------------------------------
# -- Conjuntos de entrenamiento y prueba --
P <- ncol(datos) - 1 # Numero de variables
N <- nrow(datos) # Numero de observaciones
cat("Contamos con ", P, " variables predictoras y ", N, " observaciones\n")

X <- as.matrix(datos[, -which(names(datos) %in% c("Chance.of.Admit"))])
Y <- as.numeric(datos$Chance.of.Admit)

train_samples_index <- which(sample(c(TRUE, FALSE), size = N, replace = TRUE, prob = c(0.8, 0.2)))
Xtrain <- X[train_samples_index,]
Xtest <- X[-train_samples_index,]
Ytrain <- Y[train_samples_index]
Ytest <- Y[-train_samples_index]
# ------------------------------------------
# -- Modelo lineal --
lm_modelo <- lm(Chance.of.Admit ~ . - Chance.of.Admit, data = datos, subset = train_samples_index)
summary(lm_modelo)
preds_lm <- predict(lm_modelo, data.frame(Xtest))
Metricas_lm <- CalcularR2(preds_lm, Ytest, N, P)
cat("-- Resultados de modelo lineal -- \nR2 = ", Metricas_lm$R2, "\nR2 ajustado = ", Metricas_lm$R2ajust, "\nMSE = ", Metricas_lm$MSE, "\nMAE = ", Metricas_lm$MAE, "\n")

VisualizarPredReal(preds_lm, Ytest, main = "Gráfica predicción-observación de modelo lineal")

# -- Modelo Ridge con VC --
# Valores lambda propuestos para el modelado
lambda_vals <- 10^seq(-3, 0, length.out = 1024)

ridge_modelo <- cv.glmnet(Xtrain, Ytrain, alpha = 0, lambda = lambda_vals)
plot(ridge_modelo)
best_coefs_ridge <- coef(ridge_modelo, s = ridge_modelo$lambda.min)
preds_ridge <- predict(ridge_modelo, newx = Xtest, s = ridge_modelo$lambda.min)
Metricas_ridge <- CalcularR2(preds_ridge, Ytest, N, P)
cat("-- Resultados de modelo Ridge -- \nR2 = ", Metricas_ridge$R2, "\nR2 ajustado = ", Metricas_ridge$R2ajust, "\nMSE = ", Metricas_ridge$MSE, "\nMAE = ", Metricas_ridge$MAE, "\n")

VisualizarPredReal(preds_ridge, Ytest, main = "Gráfica predicción-observación de modelo Ridge")

# -- Modelo Lasso con VC --
lasso_modelo <- cv.glmnet(Xtrain, Ytrain, alpha = 1, lambda = lambda_vals)
plot(lasso_modelo)
best_coefs_lasso <- coef(lasso_modelo, s = lasso_modelo$lambda.min)
preds_lasso <- predict(lasso_modelo, newx = Xtest, s = lasso_modelo$lambda.min)
Metricas_lasso <- CalcularR2(preds_lasso, Ytest, N, P)
cat("-- Resultados de modelo Lasso -- \nR2 = ", Metricas_lasso$R2, "\nR2 ajustado = ", Metricas_lasso$R2ajust, "\nMSE = ", Metricas_lasso$MSE, "\nMAE = ", Metricas_lasso$MAE, "\n")

VisualizarPredReal(preds_lasso, Ytest, main = "Gráfica predicción-observación de modelo Lasso")

# ------------------------------------------
# -- Dispersion de metricas para los modelos propuestos --
num_reps <- 100

Metricas_reps_lm <- data.frame(matrix(nrow = num_reps, ncol = 4))
Metricas_reps_ridge <- data.frame(matrix(nrow = num_reps, ncol = 4))
Metricas_reps_lasso <- data.frame(matrix(nrow = num_reps, ncol = 4))

for(num in 1:num_reps){
    rep_train_samples_index <- which(sample(c(TRUE, FALSE), size = N, replace = TRUE, prob = c(0.8, 0.2)))
    X_rep_train <- X[rep_train_samples_index,]
    X_rep_test <- X[-rep_train_samples_index,]
    Y_rep_train <- Y[rep_train_samples_index]
    Y_rep_test <- Y[-rep_train_samples_index]

    # Modelo lineal
    lm_rep_model <- lm(Chance.of.Admit ~ . - Chance.of.Admit, data = datos, subset = rep_train_samples_index)
    preds_rep_lm <- predict(lm_rep_model, data.frame(X_rep_test))
    Metricas_rep_lm <- CalcularR2(preds_rep_lm, Y_rep_test, N, P)

    # Modelo Ridge
    ridge_rep_model<- cv.glmnet(X_rep_train, Y_rep_train, alpha = 0, lambda = lambda_vals)
    preds_rep_ridge <- predict(ridge_rep_model, newx = X_rep_test, s = ridge_rep_model$lambda.min)
    Metricas_rep_ridge <- CalcularR2(preds_rep_ridge, Y_rep_test, N, P)

    # Modelo Lasso
    lasso_rep_model<- cv.glmnet(X_rep_train, Y_rep_train, alpha = 1, lambda = lambda_vals)
    preds_rep_lasso <- predict(lasso_rep_model, newx = X_rep_test, s = lasso_rep_model$lambda.min)
    Metricas_rep_lasso <- CalcularR2(preds_rep_lasso, Y_rep_test, N, P)

    Metricas_reps_lm[num, ] <- unlist(Metricas_rep_lm)
    Metricas_reps_ridge[num, ] <- unlist(Metricas_rep_ridge)
    Metricas_reps_lasso[num, ] <- unlist(Metricas_rep_lasso)

    print(num)
}
Metricas_reps_lm$fuente <- "Lineal"
Metricas_reps_ridge$fuente <- "Ridge"
Metricas_reps_lasso$fuente <- "Lasso"

Metricas_reps_combinadas <- bind_rows(Metricas_reps_lm, Metricas_reps_ridge, Metricas_reps_lasso)
names(Metricas_reps_combinadas)[1:4] <- c("R2", "R2ajust", "MSE", "MAE")

ggplot(Metricas_reps_combinadas, aes(x = fuente, y = MSE, fill = fuente)) +
geom_boxplot() +
theme_minimal() +
labs(title = "Boxplots de MSE por modelo",
    x = "Modelo",
    y = "MSE")

ggplot(Metricas_reps_combinadas, aes(x = fuente, y = MAE, fill = fuente)) +
geom_boxplot() +
theme_minimal() +
labs(title = "Boxplots de MAE por modelo",
    x = "Modelo",
    y = "MAE")

ggplot(Metricas_reps_combinadas, aes(x = fuente, y = R2, fill = fuente)) +
geom_boxplot() +
theme_minimal() +
labs(title = "Boxplots de R2 por modelo",
    x = "Modelo",
    y = "R2")

ggplot(Metricas_reps_combinadas, aes(x = fuente, y = R2ajust, fill = fuente)) +
geom_boxplot() +
theme_minimal() +
labs(title = "Boxplots de R2 ajustada por modelo",
    x = "Modelo",
    y = "R2 ajustada")