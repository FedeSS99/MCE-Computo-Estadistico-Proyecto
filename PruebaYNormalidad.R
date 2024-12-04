source("./CargaDatos.R")

Y <- as.vector(datos$Chance.of.Admit)
LogY <- log10(max(Y + 1) - Y)
SqrtY <- sqrt(max(Y + 1) - Y)
InvY <- 1.0 / (max(Y + 1) - Y)

shapiro.test(Y)
shapiro.test(LogY)
shapiro.test(SqrtY)
shapiro.test(InvY)

car::qqPlot(Y)
car::qqPlot(LogY)
car::qqPlot(SqrtY)
car::qqPlot(InvY)

hist(Y)
hist(LogY)
hist(SqrtY)
hist(InvY)