

# pdf('Fig4.pdf', width=10, height=5)
png('Fig4.png')
par(mfrow = c(1, 2))

df <- read.csv('C:/Users/Milton/Documents/Doc/Empatica-Project-ALAS-main-v2/UMateria/EEG_Z39S39N100_R2.csv')

x2 <- df
bands <- c('Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
for (band in bands) {
     # Create new averaged column
     x2[, band] <- (df[[paste0(band, '_C4')]] + df[[paste0(band, '_C3')]]) / 2
}
x2 <- x2[, -c(1:10)]
bad_ids <- c(6, 20, 19, 1, 5, 11, 12, 30, 31, 36)
x3 <- x2[!(x2$Subject %in% bad_ids),]

x3['Engagement'] <- abs(x3$Beta / (x3$Alpha + x3$Theta))
x3['Fatigue'] <- abs(x3$Theta / x3$Alpha)
x3['Excitement'] <- abs(x3$Beta / x3$Alpha)

mmean <- t(rbind(tapply(x3$Engagement, x3$Group, mean),
                 tapply(x3$Fatigue, x3$Group, mean),
                 tapply(x3$Excitement, x3$Group, mean)))
msd <- t(rbind(tapply(x3$Engagement, x3$Group, sd),
               tapply(x3$Fatigue, x3$Group, sd),
               tapply(x3$Excitement, x3$Group, sd)))/sqrt(700*6)
cols <- c('#808080', '#6baed6', '#2ca25f')

# png('validation_index.png')
# pdf('Tec/UMateria/validation_index.pdf', width=5, height=5)
bp <- barplot(mmean, beside = TRUE, names.arg = c('Engagement', 'Fatigue','Excitement'),
              ylab = 'Index', axes = FALSE, col = cols, main = 'A)',
              ylim = c(0, 5))
legend('topleft', legend = c('Control', 'Index', 'Model'),
       fill = cols, bty = 'n')
axis(2, at = c(0, 1, 2, 3, 4), labels = c(0, 1, 2, 3, 4))
arrows(bp, mmean, bp, mmean + msd, code = 3, angle = 90, length = 0.15)
text(bp, mmean + msd + 0.15, labels = round(mmean, 2))

add_segments <- function(x0, x1, y){
     l <- 0.05
     segments(x0 = x0, x1 = x1, y0 = y, y1 = y, lwd = 1, col = 1)
     segments(x0, y - l, x0, y, lwd = 1)
     segments(x1, y - l, x1, y, lwd = 1)}

add_segments(1.5, 2.5, 2.1); text(2, 2.25, '***', cex = 2)
add_segments(1.5, 3.5, 2.4); text(2.5, 2.55, '***', cex = 2)
# add_segments(5.5, 6.5, 3); text(6, 3.1, '*', cex = 2)
add_segments(5.5, 7.5, 3.7); text(6.5, 3.85, '*', cex = 2)
#add_segments(9.5, 10.5, 2.85); text(10, 2.95, '*', cex = 2)
#add_segments(10.5, 11.5, 4.3); text(11, 4.4, '*', cex = 2)
add_segments(9.5, 11.5, 4.4); text(10.5, 4.55, '**', cex = 2)
legend('top', c('* p < 0.05', '** p < 0.01', '*** p < 0.001'), bty = 'n', cex=0.85)
# dev.off()

# tes <- with(x3, kruskal.test(Engagement ~ Group))

TukeyHSD(with(x3, aov(Engagement ~ Group)))
TukeyHSD(with(x3, aov(Fatigue ~ Group)))
TukeyHSD(with(x3, aov(Excitement ~ Group)))

# Theta
# 11 = 2.26, 14 = 2.36, 15 = 1.8

# Alpha
# 11 = 1.54, 5 = 2.21

# Beta
# 11 = 2.09 

#######

# 2  3  4  7  8  9 13 14 15 16 17 18 32 33 34 37
# 6, 6, 4

smoo <- function(data){
     smoothing_span <- 6
     smoothed_data <- smooth.spline(seq_along(data), data, spar = 1 - 1 / smoothing_span)
     smoothed_values <- predict(smoothed_data)
     return(smoothed_values$y)}

feature <- 'Engagement'
a <- smoo(x3[x3$Subject == 3, feature])
b <- smoo(x3[x3$Subject == 34, feature])
c <- smoo(x3[x3$Subject == 17, feature])
cols <- c('#808080', '#6baed6', '#2ca25f')

# png('conti_validation.png')
# pdf('Tec/UMateria/conti_validation.pdf', width=5, height=5)
plot(1:length(a)/60 + 90/60, a, type = 'l', ylim = c(0, 1.1), axes = FALSE,
     main = 'B) Subjects: #3, #17, #24', col = cols[1], xlim = c(3, 13),
     lwd = 2, ylab = 'Engagement index', xlab = 'Time (min)')
lines(1:length(b)/60 + 90/60, b, lwd = 2, col = cols[2])
lines(1:length(c)/60 + 90/60, c, lwd = 2, col = cols[3])
axis(1)
axis(2, at = c(0, 0.5, 1), labels = c(0, 0.5, 1))
legend('topleft', legend = c('Control', 'Index', 'Model'), lwd = 4, col = cols,
       bty = 'n')
abline(h = 0.5, lty = 2, col = 'grey80')

dev.off()







######## OTHER PLOT ########

# Bandpower analysis
mmean <- t(rbind(tapply(x3$Delta, x3$Group, mean),
                 tapply(x3$Theta, x3$Group, mean),
                 tapply(x3$Alpha, x3$Group, mean),
                 tapply(x3$Beta, x3$Group, mean),
                 tapply(x3$Gamma, x3$Group, mean)))

bp <- barplot(mmean, beside = TRUE, names.arg = c('Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'),
              ylab = 'Index', axes = FALSE, col = c('grey80', 'grey20'), ylim = c(-0.25, 2.25))
legend('topleft', legend = c('Control', 'Experimental'),
       fill = c('grey80', 'grey20'), bty = 'n')
axis(2, at = c(0, 1, 2), labels = c(0, 1, 2))

msd <- t(rbind(tapply(x3$Delta, x3$Group, sd),
               tapply(x3$Theta, x3$Group, sd),
               tapply(x3$Alpha, x3$Group, sd),
               tapply(x3$Beta, x3$Group, sd),
               tapply(x3$Gamma, x3$Group, sd)))/sqrt(900)

arrows(bp, mmean, bp, mmean + msd, code = 3, angle = 90, length = 0.15)
text(bp, mmean + msd + 0.1, labels = round(mmean, 2))

tapply(x2$Alpha, x2$Subject, mean)

# print(df)
# df2 <- df[(df$ID == 1) & (df$Take == 1),]

plot(df[(df$ID == 2) & (df$Take == 1), variable], ylim = c(0, 1))
points(df[(df$ID == 1) & (df$Take == 2), variable], col = 'red')

barplot(t(as.matrix(tapply(df$Theta - df$Alpha, list(df$ID, df$Take), mean))), beside=TRUE)

barplot(t(as.matrix(tapply(df$Fatigue, list(df$ID, df$Take), mean))), beside=TRUE)

plot(df[(df$ID == 2) & (df$Take == 1), variable], ylim = c(0, 1))
points(df[(df$ID == 1) & (df$Take == 2), variable], col = 'red')

smoo <- function(data){
     smoothing_span <- 3
     smoothed_data <- smooth.spline(seq_along(data), data, spar = 1 - 1 / smoothing_span)
     smoothed_values <- predict(smoothed_data)
     return(smoothed_values$y)
}

variable <- 'Engagement'
sub <- 3

x1 <- smoo(df[(df$ID == sub) & (df$Take == 2), variable])
x2 <- smoo(df[(df$ID == sub) & (df$Take == 1), variable])

th <- mean(mean(x1), mean(x2))
png('engagement.png')
plot(x1, type = 'l', ylab = variable, xlab = 'Time (s)', xlim = c(20, 80),
     ylim = c(0, max(max(x1), max(x2))), axes = FALSE, main = 'Subject #3')
lines(x2, col = 'red')
axis(1)
axis(2)
legend('topright', legend = c('Control', 'Immersive'), lty = 1, col = c(1, 2), bty = 'n')
dev.off()

meng <- t(as.matrix(tapply(df$Engagement, list(df$ID, df$Take), mean)))
meng <- meng[nrow(meng):1,]

# png('eng1.png')
pdf('bareng.pdf')
barplot(meng, beside=TRUE, col = c('grey80', 'grey20'), ylab = 'Engagement')
legend(fill = c('grey80', 'grey20'), 'topleft', legend = c('Non-Immersive', 'Immersive'), bty = 'n')
dev.off()

mfat <- t(as.matrix(tapply(df$Fatigue, list(df$ID, df$Take), mean)))
mfat <- mfat[nrow(mfat):1,]

png('fat1.png')
barplot(t(as.matrix(tapply(df$Fatigue, list(df$ID, df$Take), mean))), beside=TRUE,
        col = c('grey20', 'grey80'), ylab = 'Fatigue')
legend(fill = c('grey20', 'grey80'), 'topright', legend = c('Immersive', 'Control'), bty = 'n')
dev.off()

mrel <- t(as.matrix(tapply(df$Relaxation, list(df$ID, df$Take), mean)))
mrel <- mrel[nrow(mrel):1,]


### Other plot ###

pdf('eeg_emo.pdf')
mmean <- t(rbind(tapply(df$Engagement, list(df$Group, df$Take), mean)[,1],
                 tapply(df$Fatigue, list(df$Group, df$Take), mean)[,1],
                 tapply(df$Excitement, list(df$Group, df$Take), mean)[,1]))
mmean <- mmean[nrow(mmean):1,]

bp <- barplot(mmean, beside = TRUE, names.arg = c('Engagement', 'Fatigue', 'Excitement'),
              ylab = 'Index', axes = FALSE, col = c('#377EB8', '#E69F00'), ylim = c(0, 4.5))
legend('topleft', legend = c('Sadness', 'Hapiness'),
       fill = c('#377EB8', '#E69F00'), bty = 'n')
axis(2, at = c(0, 1, 2, 3, 4), labels = c(0, 1, 2, 3, 4))

msd <- t(rbind(tapply(df$Engagement, list(df$Group, df$Take), mean)[,1],
               tapply(df$Fatigue, list(df$Group, df$Take), mean)[,1],
               tapply(df$Excitement, list(df$Group, df$Take), mean)[,1]))/6
msd <- msd[nrow(msd):1,]

arrows(bp, mmean, bp, mmean + msd, code = 3, angle = 90, length = 0.15)
text(bp, mmean + msd + 0.15, labels = round(mmean, 2))

add_segments <- function(x0, x1, y){
     l <- 0.05
     segments(x0 = x0, x1 = x1, y0 = y, y1 = y, lwd = 1, col = 1)
     segments(x0, y - l, x0, y, lwd = 1)
     segments(x1, y - l, x1, y, lwd = 1)}

add_segments(7.5, 8.5, 4.25)
text(8, 4.35, '*', cex = 2)
legend('top', '* p < 0.05', bty = 'n', cex = 1.25)
dev.off()