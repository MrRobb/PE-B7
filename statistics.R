# Set current working directory where the folder is
browseURL("https://i.stack.imgur.com/7A467.jpg")

# Load data
layer1data <- read.csv("results/1layer/1layer_5000iterations_1step.csv")
layer3data <- read.csv("results/3layer/3layer_5000iterations_1step.csv")
layer5data <- read.csv("results/5layer/5layer_5000iterations_1step.csv")

# Plot data

png(file = "resultslayersdata.png")
plot(layer1data$Test.Accuracy, type = "l", col = "red", xlab = "Iteraciones", ylab = "Probabilidad de acierto", 
     main = "Precision del test")
lines(layer3data$Test.Accuracy, type = "l", col = "blue")
lines(layer5data$Test.Accuracy, type = "l", col = "green")
dev.off()

png(file = "results/layersdatazoom.png")
plot(layer1data$Test.Accuracy, type = "l", col = "red", xlab = "Iteraciones", ylab = "Probabilidad de acierto", 
     main = "Precision del test", xlim = c(4900, 5000), ylim = c(0.98, 1.0))
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