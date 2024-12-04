CalcularMetricas <- function(pred, real, n, p){
    RSS <- sum((pred - real)^2)
    MSE <- mean(RSS)
    MAE <- mean(abs(pred - real))

    R2 <- cor(real, pred)^{2}
    R2ajust <- 1.0 - (1.0 - R2)*((n - 1)/(n - p - 1))

    return(list(R2 = R2, R2ajust = R2ajust, MSE = MSE, MAE = MAE))
}


VisualizarPredReal <- function(pred, real, main){
    plot(pred, real, type = "p", pch = 19, col = "blue", xlab = "Valore predichos", ylab = "Valores reales", main = main)
    abline(a = 0.0, b = 1.0, col = "red")
}
