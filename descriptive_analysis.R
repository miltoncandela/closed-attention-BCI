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
x <- x[,c(colnames(x) != 'Subject')]
x2 <- df

# y =  (1 - ((y - 0.2)/(0.7-0.2))) * 100
# y = (1 - ((y - min(y))/(max(y)-min(y)))) * 100

# Step 3: Loop through specified bands to calculate averages and drop original columns
bands <- c('Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
for (band in bands) {
     # Create new averaged column
     x2[, band] <- (df[[paste0(band, '_C4')]] + df[[paste0(band, '_C3')]]) / 2
}
x2 <- x2[, -c(1:10)]

#for (band in bands) {
#     x2[, paste0(band, '2')] <- x2[, band] * x2[, band]
#}

for (band in bands) {
     x2 <- x2[abs(x2[,band]) < 3,]
}

y <- x2['RT']
x2 <- x2[,-1]
y = (1 - ((y - min(y))/(max(y)-min(y)))) * 100
# x[,'RT'] <- (1 - ((x$RT - min(x$RT))/(max(x$RT)-min(x$RT)))) * 100

#combinations <- combn(names(x2), 2, simplify = FALSE)

#for (combo in combinations){
#     feature_name <- paste(combo, collapse = '_')
#     x2[feature_name] <- x2[combo[1]] * 1/x2[combo[2]]
#}

x2['RT'] <- y
# x2 <- x2[x2$RT > 0.3,]

col_band <- c('darkgrey', 'mediumslateblue', 'lightcoral', 'cornflowerblue', 'wheat')

plot(x2[,'Theta'], x2[,'RT'], xlab = 'Theta (μV^2/Hz)', ylab = 'Engagement Score',
     col = col_band[2], pch = 20, cex = 2, xlim = c(-0.5, 0.1), ylim = c(0, 100),
     axes = FALSE)
a <- x2[,'Theta']
model <- lm(x2[,'RT'] ~ a + 0)
abline(model, lwd = 2)
axis(1)
axis(2)


plot(x2[,'Alpha'] + 2, x2[,'RT'], xlab = 'Alpha (μV^2/Hz)', ylab = 'Engagement Score',
     col = col_band[3], pch = 20, cex = 2, xlim = c(1, 3), ylim = c(0, 100))
a <- x2[,'Alpha'] + 2
model <- lm(x2[,'RT'] ~ a + 0)
abline(model)

plot(x2[,'Beta'] + 2, x2[,'RT'], xlab = 'Beta (μV^2/Hz)', ylab = 'Engagement Score',
     col = col_band[4], pch = 20, cex = 2, xlim = c(1, 3), ylim = c(0, 100), axes = FALSE)
a <- x2[,'Beta'] + 2
model <- lm(x2[,'RT'] ~ a + 0)
abline(model, lwd = 2)






