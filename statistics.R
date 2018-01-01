# Set current working directory where the folder is
browseURL("https://i.stack.imgur.com/7A467.jpg")

# Load data
layer1data <- read.csv("results/1layer/1layer_5000iterations_1step.csv")
layer3data <- read.csv("results/3layer/3layer_5000iterations_1step.csv")
layer5data <- read.csv("results/5layer/5layer_5000iterations_1step.csv")

# Plot data

# png(file = "layer1data.png")
plot(layer1data$Test.Accuracy, type = "l", col = "red", xlab = "Iteraciones", ylab = "Probabilidad de acierto", 
     main = "Precision del test")
lines(layer3data$Test.Accuracy, type = "l", col = "blue")
lines(layer5data$Test.Accuracy, type = "l", col = "green")
# dev.off()
