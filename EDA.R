library(dplyr)
library(ggplot2)
library(reshape2)
library(car)
library(heatmaply)


source("./CargaDatos.R")

# Transformación logarítmica en la variable objetivo
datos$Chance.of.Admit <- log(datos$Chance.of.Admit + 1)

# ------------------------------------------
# -- Revision de columnas con NAs --
ColNAs <- colSums(is.na(datos))
ColNAs

# ------------------------------------------
# Distribucion de las variables
numericas <- datos[, sapply(datos, is.numeric)]

for (var in colnames(numericas)) {
  p <- ggplot(datos, aes(x = .data[[var]])) +
    geom_histogram(bins = 30, fill = "steelblue", color = "white") +
    theme_minimal() +
    labs(title = paste("Distribución de", var), x = var, y = "Frecuencia")
  
  print(p) 
}

# ------------------------------------------
# Revisar correlación
cor_matrix <- cor(datos[, -which(names(datos) == "Chance.of.Admit")])

cor_data <- melt(cor_matrix)
ggplot(cor_data, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  theme_minimal() +
  labs(title = "Matriz de correlación", x = "Variables", y = "Variables") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
heatmaply(cor_matrix, main = "Mapa de Calor de Correlaciones")

# ------------------------------------------
# VIF
full_model <- lm(Chance.of.Admit ~ ., data = datos)
vif_values <- vif(full_model)
print(vif_values)

# Visualizar
vif_df <- data.frame(Variable = names(vif_values), VIF = vif_values)
ggplot(vif_df, aes(x = reorder(Variable, -VIF), y = VIF)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  theme_minimal() +
  labs(title = "Factor de inflación de la varianza (VIF)",
       x = "Variable",
       y = "VIF") +
  geom_hline(yintercept = 5, color = "red", linetype = "dashed")