# Set current working directory where the folder is
browseURL("https://i.stack.imgur.com/7A467.jpg")

# Load data
layer1Data <- read.csv("results/1layer/1layer_5000iterations_1step.csv")
layer3Data <- read.csv("results/3layer/3layer_5000iterations_1step.csv")
layer5Data <- read.csv("results/5layer/5layer_5000iterations_1step.csv")

# Plot data

png(file = "images/layersdata.png", width = 10, height = 10, units = 'in', res = 100)
plot(layer1Data$Test.Accuracy, 
     type = "l", col = "red", 
     xlab = "Iteraciones", 
     ylab = "Probabilidad de acierto", 
     main = "Precision")
lines(layer3Data$Test.Accuracy, type = "l", col = "blue")
lines(layer5Data$Test.Accuracy, type = "l", col = "green")
dev.off()

png(file = "images/layersdatazoom.png", width = 10, height = 10, units = 'in', res = 100)
plot(layer1Data$Test.Accuracy, 
     type = "l", col = "red", 
     xlab = "Iteraciones", 
     ylab = "Probabilidad de acierto", 
     main = "Precision", 
     xlim = c(4900, 5000), ylim = c(0.98, 1.0))
lines(layer3Data$Test.Accuracy, type = "l", col = "blue")
lines(layer5Data$Test.Accuracy, type = "l", col = "green")
dev.off()

"layer1Data"
summary(layer1Data$Iterations)
summary(layer1Data$Training.Accuracy)
summary(layer1Data$Training.Loss)
summary(layer1Data$Test.Accuracy)
summary(layer1Data$Test.Loss)

"layer3Data"
summary(layer3Data$Iterations)
summary(layer3Data$Training.Accuracy)
summary(layer3Data$Training.Loss)
summary(layer3Data$Test.Accuracy)
summary(layer3Data$Test.Loss)

"layer5Data"
summary(layer5Data$Iterations)
summary(layer5Data$Training.Accuracy)
summary(layer5Data$Training.Loss)
summary(layer5Data$Test.Accuracy)
summary(layer5Data$Test.Loss)

n1 = length(layer1Data$Training.Accuracy);
n3 = length(layer3Data$Training.Accuracy);
n5 = length(layer5Data$Training.Accuracy);

s1 = sd (layer1Data$Training.Accuracy)
s3 = sd (layer3Data$Training.Accuracy)
s5 = sd (layer5Data$Training.Accuracy)

m1 = mean (layer1Data$Training.Accuracy)
m3 = mean (layer3Data$Training.Accuracy)
m5 = mean (layer5Data$Training.Accuracy)

z13 = (m1-m3)/(sqrt((s1^2)/n1)+((s3^2)/n3)); z13
z35 = (m3-m5)/(sqrt((s3^2)/n3)+((s5^2)/n5)); z35

confi = 0.999
zalfa = qnorm(confi, mean = 0, sd = 1)

if (zalfa < abs(z13))
{
  cat("Rechazamos Ho de 1 y 3 -> 3 es mejor que 1, con una confianza del", confi*100, "%")
} else print("No podemos rechazar Ho de 1 y 3")   


if (zalfa < abs(z35))
{
  cat("Rechazamos Ho de 3 y 5 -> 5 es mejor que 3 con una confianza del", confi*100,"%")
} else print("No podemos rechazar Ho de 3 y 5")


timingData <- read.csv("results/timing.csv")
func <- lm(timingData$Tiempo..segundos..Y~timingData$N..mero.de.Capas.X)
summary(func)
png(file = "images/timing.png", width = 10, height = 10, units = 'in', res = 100)
plot(timingData, col = "blue", main = "Tiempo de entrenamiento", xlab = "Capas", ylab = "Segundos")
abline(func)
dev.off()

cat("Y = ", func$coefficients[2], "· X +", func$coefficients[1])

cat("Tiempo de entrenamiento red 2 layers =", func$coefficients[1] + func$coefficients[2] * 2, "segundos")
cat("Tiempo de entrenamiento red 4 layers =", func$coefficients[1] + func$coefficients[2] * 4, "segundos")
