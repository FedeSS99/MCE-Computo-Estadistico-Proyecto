library(dplyr)


datos_burnout <- read.csv("./data/train.csv")

# Removemos la columna Employee.ID
datos_burnout <- datos_burnout[, -which(names(datos_burnout) == "Employee.ID")]
head(datos_burnout)

# Revisamos el numero de NAs por cada variable
colSums(is.na(datos_burnout))

# Removemos filas con Burn.Rate con NA
datos_burnout <- datos_burnout[-which(is.na(datos_burnout$Burn.Rate)), ]
colSums(is.na(datos_burnout))

#lm_burnout <- lm(Burn.Rate ~ . - Burn.Rate - Employee.ID - Date.of.Joining, data = datos_burnout)
#summary(lm_burnout)
