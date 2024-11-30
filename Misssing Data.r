# Se remueven los datos faltantes de la variable predictora, burn Rate
train_data <- train_data[!is.na(train_data$`Burn Rate`), ]
colSums(is.na(train_data))


# Revisamos patrones de los datos faltantes de las variables Mental Fatigue Score y Resource Allocation
