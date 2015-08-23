## Question 1: Have total emissions from PM2.5 decreased in the United States from 1999 to 2008?
library(dplyr)
## read and extract the total emission data from PM 2.5
NEI <- readRDS("summarySCC_PM25.rds")
totalemission <- summarise(group_by(NEI, year), mtEmissions = sum(Emissions)/10^6)

## plot into the png file
png(file = "plot1.png", width = 480, height = 480)
plot(totalemission, type = "l", main = "Total PM2.5 Emissions in the U.S. from 1999 to 2008",
     xlab = "Year", ylab = "Emissions (Million Tons)")

## close the file device
dev.off()

## Question 2: Have total emissions from PM2.5 decreased in the Baltimore City, Maryland (fips == 24510) from 1999 to 2008? 
bcNEI <- filter(NEI, fips == "24510")

## calculate the total emissions by year
bcEmission <- summarise(group_by(bcNEI, year), Emissions = sum(Emissions))

## plot into the png file
png(file = "plot2.png", width = 480, height = 480)
plot(bcEmission, type = "l", main = "Total PM2.5 Emissions in the Baltimore City (1999 - 2008)",
     xlab = "Year", ylab = "Emissions (Tons)")

## close the file device
dev.off()

## Question 3: Of the four types of sources indicated by the type (point, nonpoint, onroad, nonroad) variable, 
## which of these four sources have seen decreases in emissions from 1999–2008 for Baltimore City? Which have seen 
## increases in emissions from 1999–2008? Use the ggplot2 plotting system to make a plot answer this question.
## Upload a PNG file containing your plot addressing this question.
bcNEI <- filter(NEI, fips == "24510")

## calculate the total emissions by type and by year
bcEmission <- summarise(group_by(bcNEI, type, year), Emissions = sum(Emissions))

## use ggplot2 to plot into the png file
png(file = "plot3.png", width = 720, height = 480)
g <- ggplot(bcEmission, aes(year, Emissions))
g <- g + geom_point()
g <- g + facet_grid(facets = . ~ type)
g <- g + geom_smooth(method = "lm", se = FALSE, col = "steelblue")
g <- g + labs(x = "Year") + labs(y = "Emissions")
g <- g + labs(title = "PM2.5 Emissions in the Baltimore City by Type (1999 - 2008)")
print(g)

## close the file device
dev.off()

## Question 4:Across the United States, how have emissions from coal combustion-related sources changed from 1999–2008?
SCC <- readRDS("Source_Classification_Code.rds")

## find the SCC for coal combustion - related sources
SCC <- mutate(SCC, Long.Name = paste(SCC.Level.One, SCC.Level.Two, SCC.Level.Three, 
                                     SCC.Level.Four, sep = ""))
coSCC <- SCC[grep("*coal*", SCC$Long.Name, ignore.case = TRUE), ] 
coSCC <- coSCC[grep("*combustion*", coSCC$Long.Name, ignore.case = TRUE),]
coNEI <- NEI[NEI$SCC %in% as.character(coSCC$SCC),]

## calculate the total emissions by year
coEmission <- summarise(group_by(coNEI, year), sum(Emissions))

## plot into the png file
png(file = "plot4.png", width = 480, height = 480)
plot(coEmission, type = "l", main = "Total PM2.5 Emissions (Coal Combustion-related)",
     xlab = "Year (1999 - 2008)", ylab = "Emissions (Tons)")

## close the file device
dev.off()
## Question 5: How have emissions from motor vehicle sources changed from 1999–2008 in Baltimore City?
SCC <- readRDS("Source_Classification_Code.rds")

## find the SCC for motor vehicle sources in Baltimore City
SCC <- mutate(SCC, Long.Name = paste(SCC.Level.One, SCC.Level.Two, SCC.Level.Three, 
                                     SCC.Level.Four, sep = ""))
moSCC <- SCC[grep("*motor*", SCC$Long.Name, ignore.case = TRUE), ] 
moSCC <- moSCC[grep("*vehicle*", moSCC$Long.Name, ignore.case = TRUE),]
bcmoNEI <- NEI[NEI$fips == "24510" & NEI$SCC %in% as.character(moSCC$SCC) ,]

## calculate the total emissions by year
bcmoEmission <- summarise(group_by(bcmoNEI, year), sum(Emissions))

## plot into the png file
png(file = "plot5.png", width = 480, height = 480)
plot(bcmoEmission, type = "l",
     main = "PM2.5 Emissions from moto vehicle in Baltimore City",
     xlab = "Year (1999 - 2008)", ylab = "Emissions (Tons)")

## close the file device
dev.off()

## question 6: Compare emissions from motor vehicle sources in Baltimore City with emissions from motor vehicle 
## sources in Los Angeles County, California (fips == 06037). Which city has seen greater changes over time in motor
## vehicle emissions?
## find the SCC for motor vehicle sources
SCC <- mutate(SCC, Long.Name = paste(SCC.Level.One, SCC.Level.Two, SCC.Level.Three, 
                                     SCC.Level.Four, sep = ""))
moSCC <- SCC[grep("*motor*", SCC$Long.Name, ignore.case = TRUE), ] 
moSCC <- moSCC[grep("*vehicle*", moSCC$Long.Name, ignore.case = TRUE),]
moNEI <- NEI[NEI$SCC %in% as.character(moSCC$SCC) ,]

## extract data from Baltimore city and Los Angeles Country
moNEI <- filter(moNEI, fips == "24510" | fips == "06037")
moNEI <- mutate(moNEI, city = ifelse(fips == "24510", "Baltimore", "Los Angeles"))

## calculate the total emissions by year by city
moEmission <- summarise(group_by(moNEI, city, year), Emissions = sum(Emissions))

## plot into the png file
png(file = "plot6.png", width = 480, height = 480)
g <- qplot(year, Emissions, data = moEmission, color = city,  
           geom = c("point", "smooth"), method = "lm", se = FALSE)
g <- g + labs(x = "Year (1999 - 2008)") + labs(y = "Emissions (Tons)")
g <- g + labs(title = "PM2.5 Emissions from moto vehicle in Baltimore vs. Los Angeles")
print(g)

## close the file device
dev.off()
