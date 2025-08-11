df <- read.csv('C:/Users/Milton/Documents/R/BRAIN-R/Tec/UMateria/score_report.csv')
df <- df[-c(1:2),]
print(df)

m <- tapply(df$Score, df$Group, mean)
msd <- tapply(df$Score, df$Group, sd)/sqrt(10)
cols <- c('#808080', '#6baed6', '#2ca25f')

png('performance.png')
bp <- barplot(m, ylab = 'Performance', ylim = c(50, 100), xpd = FALSE, col = cols)
arrows(bp, m, bp, m + msd, code = 3, angle = 90, length = 0.15)
text(bp, m + msd + 2, labels = round(m, 2))
dev.off()

with(df, TukeyHSD(aov(Score ~ Group)))

# # # # # 

m <- tapply(df$Engagement, df$Group, mean)
msd <- tapply(df$Engagement, df$Group, sd)/sqrt(10)
cols <- c('#808080', '#6baed6', '#2ca25f')

png('perceived.png')
bp <- barplot(m, ylab = 'Perceived engagement', ylim = c(0, 5), xpd = FALSE, col = cols)
arrows(bp, m, bp, m + msd, code = 3, angle = 90, length = 0.15)
text(bp, m + msd + 0.15, labels = round(m, 2))
dev.off()

with(df, TukeyHSD(aov(Engagement ~ Group)))

#barplot(tapply(df$Score, df$Group, mean))
#barplot(tapply(df$Engagement, df$Group, mean))