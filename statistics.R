# Set current working directory where the folder is
browseURL("https://i.stack.imgur.com/7A467.jpg")

# Load data
layer1data <- read.csv("results/1layer/1layer_5000iterations_1step.csv")
layer3data <- read.csv("results/3layer/3layer_5000iterations_1step.csv")
layer5data <- read.csv("results/5layer/5layer_5000iterations_1step.csv")

# Plot data

png(file = "resultslayersdata.png")
plot(layer1data$Test.Accuracy, 
     type = "l", col = "red", 
     xlab = "Iteraciones", 
     ylab = "Probabilidad de acierto", 
     main = "Precision del test")
lines(layer3data$Test.Accuracy, type = "l", col = "blue")
lines(layer5data$Test.Accuracy, type = "l", col = "green")
dev.off()

png(file = "results/layersdatazoom.png")
plot(layer1data$Test.Accuracy, 
     type = "l", col = "red", 
     xlab = "Iteraciones", 
     ylab = "Probabilidad de acierto", 
     main = "Precision del test", 
     xlim = c(4900, 5000), ylim = c(0.98, 1.0))
lines(layer3data$Test.Accuracy, type = "l", col = "blue")
lines(layer5data$Test.Accuracy, type = "l", col = "green")
dev.off()

"layer1data"
summary(layer1data$Iterations)
summary(layer1data$Training.Accuracy)
summary(layer1data$Training.Loss)
summary(layer1data$Test.Accuracy)
summary(layer1data$Test.Loss)

"layer3data"
summary(layer3data$Iterations)
summary(layer3data$Training.Accuracy)
summary(layer3data$Training.Loss)
summary(layer3data$Test.Accuracy)
summary(layer3data$Test.Loss)

"layer5data"
summary(layer5data$Iterations)
summary(layer5data$Training.Accuracy)
summary(layer5data$Training.Loss)
summary(layer5data$Test.Accuracy)
summary(layer5data$Test.Loss)

n1 = length(layer1data$Training.Accuracy);
n3 = length(layer3data$Training.Accuracy);
n5 = length(layer5data$Training.Accuracy);

s1 = sd (layer1data$Training.Accuracy)
s3 = sd (layer3data$Training.Accuracy)
s5 = sd (layer5data$Training.Accuracy)

m1 = mean (layer1data$Training.Accuracy)
m3 = mean (layer3data$Training.Accuracy)
m5 = mean (layer5data$Training.Accuracy)

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
