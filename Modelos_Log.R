library(dplyr)
library(tidyr)

library(glmnet)
library(pls)
library(reshape2)
library(car)

library(heatmaply)
library(ggplot2)
library(RColorBrewer)

source("./Rutinas.R")
source("./CargaDatos.R")
path_imagenes <- "./images/Modelo 2/"

# Transformación logarítmica en la variable objetivo
datos$Chance.of.Admit <- log(datos$Chance.of.Admit + 1)

# Escalado de datos de variables predictoras
VarsNames <- names(datos)
datos[, -which(names(datos) == "Chance.of.Admit")] <- scale(datos[, -which(names(datos) == "Chance.of.Admit")])
colnames(datos) <- VarsNames

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
Metricas_lm <- CalcularMetricas(preds_lm, Ytest, N, P)
cat("-- Resultados de modelo lineal -- \nR2 = ", Metricas_lm$R2, "\nR2 ajustado = ", Metricas_lm$R2ajust, "\nMSE = ", Metricas_lm$MSE, "\nMAE = ", Metricas_lm$MAE, "\n")

png(paste(path_imagenes, "Figura_PredReal_Lineal.png", sep = ""))
VisualizarPredReal(preds_lm, Ytest, main = "Gráfica predicción-observación de modelo lineal")
dev.off()

# -- Modelo Ridge con VC --
# Valores lambda propuestos para el modelado
lambda_vals <- 10^seq(-3, 0, length.out = 512)

ridge_modelo <- cv.glmnet(Xtrain, Ytrain, alpha = 0, lambda = lambda_vals)

png(paste(path_imagenes, "Figura_RidgeCV.png", sep = ""))
plot(ridge_modelo)
dev.off()

coefs_ridge <- coef(ridge_modelo, s = ridge_modelo$lambda.min)
preds_ridge <- predict(ridge_modelo, newx = Xtest, s = ridge_modelo$lambda.min)
Metricas_ridge <- CalcularMetricas(preds_ridge, Ytest, N, P)
cat("-- Resultados de modelo Ridge -- \nR2 = ", Metricas_ridge$R2, "\nR2 ajustado = ", Metricas_ridge$R2ajust, "\nMSE = ", Metricas_ridge$MSE, "\nMAE = ", Metricas_ridge$MAE, "\n")

png(paste(path_imagenes, "Figura_PredReal_Ridge.png", sep = ""))
VisualizarPredReal(preds_ridge, Ytest, main = "Gráfica predicción-observación de modelo Ridge")
dev.off()

# ------------------------------------------
# -- Modelo Lasso con VC --
lasso_modelo <- cv.glmnet(Xtrain, Ytrain, alpha = 1, lambda = lambda_vals)

png(paste(path_imagenes, "Figura_LassoCV.png", sep = ""))
plot(lasso_modelo)
dev.off()

coefs_lasso <- coef(lasso_modelo, s = lasso_modelo$lambda.min)
preds_lasso <- predict(lasso_modelo, newx = Xtest, s = lasso_modelo$lambda.min)
Metricas_lasso <- CalcularMetricas(preds_lasso, Ytest, N, P)
cat("-- Resultados de modelo Lasso -- \nR2 = ", Metricas_lasso$R2, "\nR2 ajustado = ", Metricas_lasso$R2ajust, "\nMSE = ", Metricas_lasso$MSE, "\nMAE = ", Metricas_lasso$MAE, "\n")

png(paste(path_imagenes, "Figura_PredReal_Lasso.png", sep = ""))
VisualizarPredReal(preds_lasso, Ytest, main = "Gráfica predicción-observación de modelo Lasso")
dev.off()

# ------------------------------------------
# -- Modelo PCR --
pcr_model <- pcr(Chance.of.Admit ~ . - Chance.of.Admit, data = datos, subset = train_samples_index, scale = TRUE, validation = "CV")
summary(pcr_model)
preds_pcr <- predict(pcr_model, newdata = datos[-train_samples_index,], ncomp = 3)
Metricas_pcr <- CalcularMetricas(preds_pcr, Ytest, N, P)
cat("-- Resultados de modelo PCR -- \nR2 = ", Metricas_pcr$R2, "\nR2 ajustado = ", Metricas_pcr$R2ajust, "\nMSE = ", Metricas_pcr$MSE, "\nMAE = ", Metricas_pcr$MAE, "\n")

png(paste(path_imagenes, "Figura_PCR_Comps.png", sep = ""))
validationplot(pcr_model, val.type = "MSEP")
dev.off()

png(paste(path_imagenes, "Figura_PredReal_PCR.png", sep = ""))
VisualizarPredReal(preds_pcr, Ytest, main = "Gráfica predicción-observación de modelo PCR")
dev.off()

# ------------------------------------------
# -- Modelo PLS --
pls_model <- plsr(Chance.of.Admit ~ . - Chance.of.Admit, data = datos, subset = train_samples_index, scale = TRUE, validation = "CV")
summary(pls_model)
preds_pls <- predict(pls_model, newdata = datos[-train_samples_index,], ncomp = 3)
Metricas_pls <- CalcularMetricas(preds_pls, Ytest, N, P)
cat("-- Resultados de modelo PLS -- \nR2 = ", Metricas_pls$R2, "\nR2 ajustado = ", Metricas_pls$R2ajust, "\nMSE = ", Metricas_pls$MSE, "\nMAE = ", Metricas_pls$MAE, "\n")

png(paste(path_imagenes, "Figura_PLS_Comps.png", sep = ""))
validationplot(pls_model, val.type = "MSEP")
dev.off()

png(paste(path_imagenes, "Figura_PredReal_PLS.png", sep = ""))
VisualizarPredReal(preds_pls, Ytest, main = "Gráfica predicción-observación de modelo PLS")
dev.off()

# ------------------------------------------
# -- Dispersion de metricas para los modelos propuestos --
num_reps <- 100

Metricas_reps_lm <- data.frame(matrix(nrow = num_reps, ncol = 4))
Metricas_reps_ridge <- data.frame(matrix(nrow = num_reps, ncol = 4))
Metricas_reps_lasso <- data.frame(matrix(nrow = num_reps, ncol = 4))
Metricas_reps_pcr <- data.frame(matrix(nrow = num_reps, ncol = 4))
Metricas_reps_pls <- data.frame(matrix(nrow = num_reps, ncol = 4))


mejor_modelo_lm <- list(model = NULL, R2ajust = -Inf, R2 = 0.0, MSE = 0.0, MAE = 0.0)
mejor_modelo_ridge <- list(model = NULL, R2ajust = -Inf, R2 = 0.0, MSE = 0.0, MAE = 0.0)
mejor_modelo_lasso <- list(model = NULL, R2ajust = -Inf, R2 = 0.0, MSE = 0.0, MAE = 0.0)
mejor_modelo_pcr <- list(model = NULL, R2ajust = -Inf, R2 = 0.0, MSE = 0.0, MAE = 0.0)
mejor_modelo_pls <- list(model = NULL, R2ajust = -Inf, R2 = 0.0, MSE = 0.0, MAE = 0.0)

for(num in 1:num_reps){
  rep_train_samples_index <- which(sample(c(TRUE, FALSE), size = N, replace = TRUE, prob = c(0.8, 0.2)))
  X_rep_train <- X[rep_train_samples_index,]
  X_rep_test <- X[-rep_train_samples_index,]
  Y_rep_train <- Y[rep_train_samples_index]
  Y_rep_test <- Y[-rep_train_samples_index]
  
  # Modelo lineal
  lm_rep_model <- lm(Chance.of.Admit ~ . - Chance.of.Admit, data = datos, subset = rep_train_samples_index)
  preds_rep_lm <- predict(lm_rep_model, data.frame(X_rep_test))
  Metricas_rep_lm <- CalcularMetricas(preds_rep_lm, Y_rep_test, N, P)

  if (Metricas_rep_lm$R2ajust > mejor_modelo_lm$R2ajust) {
    mejor_modelo_lm$model <- lm_rep_model
    mejor_modelo_lm$R2ajust <- Metricas_rep_lm$R2ajust
    mejor_modelo_lm$R2 <- Metricas_rep_lm$R2
    mejor_modelo_lm$MSE <- Metricas_rep_lm$MSE
    mejor_modelo_lm$MAE <- Metricas_rep_lm$MAE
  }
  
  # Modelo Ridge
  ridge_rep_model<- cv.glmnet(X_rep_train, Y_rep_train, alpha = 0, lambda = lambda_vals)
  preds_rep_ridge <- predict(ridge_rep_model, newx = X_rep_test, s = ridge_rep_model$lambda.min)
  Metricas_rep_ridge <- CalcularMetricas(preds_rep_ridge, Y_rep_test, N, P)

  if (Metricas_rep_ridge$R2ajust > mejor_modelo_ridge$R2ajust) {
    mejor_modelo_ridge$model <- ridge_rep_model
    mejor_modelo_ridge$R2ajust <- Metricas_rep_ridge$R2ajust
    mejor_modelo_ridge$R2 <- Metricas_rep_ridge$R2
    mejor_modelo_ridge$MSE <- Metricas_rep_ridge$MSE
    mejor_modelo_ridge$MAE <- Metricas_rep_ridge$MAE
    }
  
  # Modelo Lasso
  lasso_rep_model<- cv.glmnet(X_rep_train, Y_rep_train, alpha = 1, lambda = lambda_vals)
  preds_rep_lasso <- predict(lasso_rep_model, newx = X_rep_test, s = lasso_rep_model$lambda.min)
  Metricas_rep_lasso <- CalcularMetricas(preds_rep_lasso, Y_rep_test, N, P)

  if (Metricas_rep_lasso$R2ajust > mejor_modelo_lasso$R2ajust) {
    mejor_modelo_lasso$model <- lasso_rep_model
    mejor_modelo_lasso$R2ajust <- Metricas_rep_lasso$R2ajust
    mejor_modelo_lasso$R2 <- Metricas_rep_lasso$R2
    mejor_modelo_lasso$MSE <- Metricas_rep_lasso$MSE
    mejor_modelo_lasso$MAE <- Metricas_rep_lasso$MAE
  }
  
  # Modelo PCR
  pcr_rep_model <- pcr(Chance.of.Admit ~ ., data = datos, subset = rep_train_samples_index, scale = TRUE, validation = "CV")
  preds_rep_pcr <- predict(pcr_rep_model, newdata = datos[-rep_train_samples_index,], ncomp = 4)
  Metricas_rep_pcr <- CalcularMetricas(preds_rep_pcr, Y_rep_test, N, P)

  if (Metricas_rep_pcr$R2ajust > mejor_modelo_pcr$R2ajust) {
      mejor_modelo_pcr$model <- pcr_rep_model
      mejor_modelo_pcr$R2ajust <- Metricas_rep_pcr$R2ajust
      mejor_modelo_pcr$R2 <- Metricas_rep_pcr$R2
      mejor_modelo_pcr$MSE <- Metricas_rep_pcr$MSE
      mejor_modelo_pcr$MAE <- Metricas_rep_pcr$MAE
  }
  
  # Modelo PLS
  pls_rep_model <- plsr(Chance.of.Admit ~ ., data = datos, subset = rep_train_samples_index, scale = TRUE, validation = "CV")
  preds_rep_pls <- predict(pls_rep_model, newdata = datos[-rep_train_samples_index,], ncomp = 4)
  Metricas_rep_pls <- CalcularMetricas(preds_rep_pls, Y_rep_test, N, P)

  if (Metricas_rep_pls$R2ajust > mejor_modelo_pls$R2ajust) {
      mejor_modelo_pls$model <- pls_rep_model
      mejor_modelo_pls$R2ajust <- Metricas_rep_pls$R2ajust
      mejor_modelo_pls$R2 <- Metricas_rep_pls$R2
      mejor_modelo_pls$MSE <- Metricas_rep_pls$MSE
      mejor_modelo_pls$MAE <- Metricas_rep_pls$MAE
  }
  
  Metricas_reps_lm[num, ] <- unlist(Metricas_rep_lm)
  Metricas_reps_ridge[num, ] <- unlist(Metricas_rep_ridge)
  Metricas_reps_lasso[num, ] <- unlist(Metricas_rep_lasso)
  Metricas_reps_pcr[num, ] <- unlist(Metricas_rep_pcr)
  Metricas_reps_pls[num, ] <- unlist(Metricas_rep_pls)
  
  print(num)
}
cat("-- Resultados del mejor modelo LM -- \nR2 = ", mejor_modelo_lm$R2, "\nR2 ajustado = ", mejor_modelo_lm$R2ajust, "\nMSE = ", mejor_modelo_lm$MSE, "\nMAE = ", mejor_modelo_lm$MAE, "\n")

cat("-- Resultados del mejor modelo Ridge -- \nR2 = ", mejor_modelo_ridge$R2, "\nR2 ajustado = ", mejor_modelo_ridge$R2ajust, "\nMSE = ", mejor_modelo_ridge$MSE, "\nMAE = ", mejor_modelo_ridge$MAE, "\n")

cat("-- Resultados del mejor modelo Lasso -- \nR2 = ", mejor_modelo_lasso$R2, "\nR2 ajustado = ", mejor_modelo_lasso$R2ajust, "\nMSE = ", mejor_modelo_lasso$MSE, "\nMAE = ", mejor_modelo_lasso$MAE, "\n")

cat("-- Resultados del mejor modelo PCR -- \nR2 = ", mejor_modelo_pcr$R2, "\nR2 ajustado = ", mejor_modelo_pcr$R2ajust, "\nMSE = ", mejor_modelo_pcr$MSE, "\nMAE = ", mejor_modelo_pcr$MAE, "\n")

cat("-- Resultados del mejor modelo PLS -- \nR2 = ", mejor_modelo_pls$R2, "\nR2 ajustado = ", mejor_modelo_pls$R2ajust, "\nMSE = ", mejor_modelo_pls$MSE, "\nMAE = ", mejor_modelo_pls$MAE, "\n")


Metricas_reps_lm$fuente <- "Lineal"
Metricas_reps_ridge$fuente <- "Ridge"
Metricas_reps_lasso$fuente <- "Lasso"
Metricas_reps_pcr$fuente <- "PCR"
Metricas_reps_pls$fuente <- "PLS"
Metricas_reps_combinadas <- bind_rows(
  Metricas_reps_lm, 
  Metricas_reps_ridge, 
  Metricas_reps_lasso, 
  Metricas_reps_pcr, 
  Metricas_reps_pls
)
names(Metricas_reps_combinadas)[1:4] <- c("R2", "R2ajust", "MSE", "MAE")
ColoresModelos <- brewer.pal(n = 5, name = "Set1")

png(paste(path_imagenes, "Figura_MSE.png", sep = ""))
ggplot(Metricas_reps_combinadas, aes(x = fuente, y = MSE, fill = fuente)) +
  geom_boxplot(fill = ColoresModelos) +
  theme_minimal() +
  labs(title = "Boxplots de MSE por modelo",
       x = "Modelo",
       y = "MSE")
dev.off()

png(paste(path_imagenes, "Figura_MAE.png", sep = ""))
ggplot(Metricas_reps_combinadas, aes(x = fuente, y = MAE)) +
  geom_boxplot(fill = ColoresModelos) +
  theme_minimal() +
  labs(title = "Boxplots de MAE por modelo",
       x = "Modelo",
       y = "MAE")
dev.off()

png(paste(path_imagenes, "Figura_R2.png", sep = ""))
ggplot(Metricas_reps_combinadas, aes(x = fuente, y = R2, fill = fuente)) +
  geom_boxplot(fill = ColoresModelos) +
  theme_minimal() +
  labs(title = "Boxplots de R2 por modelo",
       x = "Modelo",
       y = "R2")
dev.off()

png(paste(path_imagenes, "Figura_R2ajust.png", sep = ""))
ggplot(Metricas_reps_combinadas, aes(x = fuente, y = R2ajust, fill = fuente)) +
  geom_boxplot(fill = ColoresModelos) +
  theme_minimal() +
  labs(title = "Boxplots de R2 ajustada por modelo",
       x = "Modelo",
       y = "R2 ajustada")
dev.off()

mejor_coefs_lm <- coef(mejor_modelo_lm$model)
mejor_coefs_ridge <- coef(mejor_modelo_ridge$model, s = mejor_modelo_ridge$model$lambda.min)
mejor_coefs_lasso <- coef(mejor_modelo_lasso$model, s = mejor_modelo_lasso$model$lambda.min)
mejores_coeficientes <- data.frame(
    Lineal = as.vector(mejor_coefs_lm)[2:(P + 1)],
    Ridge = as.vector(mejor_coefs_ridge)[2:(P + 1)],
    Lasso = as.vector(mejor_coefs_lasso)[2:(P + 1)]
)
mejores_coeficientes$Nombre <- names(mejor_coefs_lm)[2:(P+1)]

mejores_coeficientes <- mejores_coeficientes %>%
  pivot_longer(cols = c(Lineal, Ridge, Lasso), 
               names_to = "Modelo", 
               values_to = "Coeficiente")

png(paste(path_imagenes, "Figura_Mejores.png", sep = ""))
ggplot(data = mejores_coeficientes, aes(x = Nombre, y = Coeficiente, fill = Modelo)) +
  geom_bar(stat = "identity", position = "dodge", colour = "black") +
  scale_fill_manual(values = c( ColoresModelos[1], ColoresModelos[2], ColoresModelos[5])) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 16))
dev.off()
