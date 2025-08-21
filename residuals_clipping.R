x <- read.csv('C:/Users/Milton/PycharmProjects/ALAS/BSN/processed/EEG_Z39S39N100_R.csv')
x <- na.omit(x)

x <- x[(x[,'RT'] > 0.2) & (x[,'RT'] < 0.7),]
x <- x[,c(1:10, dim(x)[2] - 2, dim(x)[2])]

df <- data.frame(matrix(ncol = ncol(x) - 1, nrow = 0))
colnames(df) <- colnames(x)[colnames(x) != "Subject"]

for (subject in unique(x$Subject)) {
     
     df_sub <- x[x$Subject == subject, ]
     df_sub_mean <- as.data.frame(t(colMeans(df_sub[, colnames(df_sub) != "Subject"])))
     df <- rbind(df, df_sub_mean)
}

y <- df[,'RT']
x <- x[,-dim(x)[2]]
x2 <- df

bands <- c('Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
for (band in bands) {
     # Create new averaged column
     x2[, band] <- (df[[paste0(band, '_C4')]] + df[[paste0(band, '_C3')]]) / 2
}
x2 <- x2[, -c(1:10)]

for (band in bands) {
     x2 <- x2[abs(x2[,band]) < 3,]
}

y <- x2['RT']; x2 <- x2[,-1]

y <- (1 - ((y - min(y))/(max(y)-min(y)))) * 100
x2['RT'] <- y

y <- x2[,'RT']; x2 <- x2[,-1]
x3 <- x2[order(x2$RT),]

mod_propue <- with(x3, -6*Theta + 2*Alpha - 2*Beta)
print(1 - (sum((x3$RT - mod_propue*40)^2) / sum((x3$RT - mean(x3$RT))^2)))
print(mean((mod_propue*40 - x3$RT)^2))

mod_normal <- with(x3, Beta/(Alpha + Theta))
print(1 - (sum((x3$RT - mod_normal*15)^2) / sum((x3$RT - mean(x3$RT))^2)))
print(mean((mod_normal*15 - x3$RT)^2))


############### END PROCESSING ###############

pdf('Fig3.pdf', width=10, height=5)
# png('Fig3.png')
par(mfrow = c(1, 2))

# Histogram plot
# png('histrt.png')

x <- read.csv('C:/Users/Milton/PycharmProjects/ALAS/BSN/processed/EEG_Z39S39N100_R.csv')
x <- na.omit(x)
hist(x$RT, xlab = 'Reaction Time (RT) [s]', ylab = 'n', main = 'A)', xlim = c(0, 1.2))
abline(v = c(0.2, 0.7), lty = 2, col = 'red')
legend('topright', lty = 2, col = 'red', legend = 'Cut-off', bty = 'n')

# SINDy resid plot
# png('sindyresid.png')

plot(x3$RT -  mod_normal*100, col = 'red', ylab = 'Residuals', main = 'B)',
     xlab = 'Participant No.', ylim = c(-375, 375), axes=FALSE)
points(x3$RT - mod_propue*40, col = 'blue')
abline(h = 100, lty = 2)
abline(h = -100, lty = 2)
axis(1, at = seq(1, 15, 2))
axis(2, at = c(-400, -100, 0, 100, 400))
legend('topleft', legend = c('Model', 'Eng. Index'), pch = 1,
       col = c('blue', 'red'), bty = 'n')

dev.off()

############### OLD PLOTS ###############

png('sindyfig.png')
plot(x3$RT, ylim = c(0, 125), axes = FALSE, xlab = 'Participant No.',
     ylab = 'Engagement score: Min-max (1/RT)')
# points(mod_propue)
points(mod_propue*40, col = 'blue')
points(mod_normal*15, col = 'red')
legend('topleft', legend = c('True', 'SINDy', 'Eng. Index'), pch = 1,
       col = c('black', 'blue', 'red'), bty = 'n')
axis(1)
axis(2)
dev.off()

plot(x3$RT, ylim = c(0, 125), axes = FALSE, xlab = 'Participant No.',
     ylab = 'Engagement score: Min-max (1/RT)')
# points(mod_propue)

plot(x3$RT - mod_propue*40, col = 'blue')
points(x3$RT - mod_normal*100, col = 'red')


