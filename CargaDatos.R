# -- Lectura de datos y tratamiento inicial --
datos <- read.csv("./data/adm_data.csv")
datos <- datos[, -which(names(datos) == "Serial.No.")]