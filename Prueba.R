datos_burnout <- read.csv("./data/train.csv")
datos_burnout

colSums(is.na(datos_burnout))

library(misty)