# Author: Toby Anderson
# Plots for Week 2 Homework

#########################
#Plot 1 - Hot dog records
#########################

f <- file.choose()
hotdogs <- read.csv(f, sep=",", header=TRUE)

fill_colors <- c()
for (i in 1:length(hotdogs$New.record)) {
  if (hotdogs$New.record[i] == 1) {
    fill_colors <- c(fill_colors, "#821122")
  } else {
    fill_colors <- c(fill_colors, "#cccccc")
  }
}

barplot(hotdogs$Dogs.eaten,
        names.arg = hotdogs$Year
        ,col = fill_colors
        ,border = NA
        ,space = 0.3
        ,main = "Nathan's Hot Dog Eating Contest Results, 1980-2010"
        ,xlab = "Year"
        ,ylab = "Hot dogs and buns (HDB) eaten")

##############################
# Plot 2 - Hot dog by location
##############################

f <- file.choose()
hot_dog_places <- read.csv(f, sep="," ,header=TRUE)
names(hot_dog_places) <-
  c("2000", "2001", "2002", "2003", "2004",
    "2005", "2006", "2007", "2008", "2009", "2010")

hot_dog_matrix <- as.matrix(hot_dog_places)

barplot(hot_dog_matrix, border=NA, space=0.25, ylim=c(0, 200),
        xlab="Year", ylab="Hot dogs and buns (HDBs) eaten",
        main="Hot Dog Eating Contest Results, 1980-2010")

#############################
# Plot 3 - Subscribers by day
#############################

f <- file.choose()
subscribers <- read.csv(f, sep="," ,header=TRUE)

plot(subscribers$Subscribers, type="h", ylim=c(0, 30000),
     xlab="Day", ylab="Subscribers")
points(subscribers$Subscribers, pch=19, col="black")


###########################
# Plot 4 - World Population
###########################

f <- file.choose()
population <- read.csv(f, sep="," ,header=TRUE)

plot(population$Year, population$Population, type="l",
     ylim=c(0, 7000000000), xlab="Year", ylab="Population")

########################
# Plot 5 - Postage Rates
########################

f <- file.choose()
postage <- read.csv(f, sep="," ,header=TRUE)

plot(postage$Year, postage$Price, type="s",
     main="US Postage Rates for Letters, First Ounce, 1991-2010",
     xlab="Year", ylab="Postage Rate (Dollars)")


######################
# Plot 6 - LOESS curve
######################
f <- file.choose()
unemployment <- read.csv(f,sep=",", header=TRUE)
unemployment[1:10,]
plot(1:length(unemployment$Rate), unemployment$Rate)
scatter.smooth(x=1:length(unemployment$Rate),
               y=unemployment$Rate, ylim=c(0,11), degree=2, col="#CCCCCC", span=0.5)

#############################
# Plot 7 - Scatterplot Matrix
#############################
crime <- read.csv('http://datasets.flowingdata.com/crimeRatesByState2005.csv',sep=",", header=TRUE)
crime2 <- crime[crime$state != "District of Columbia",]
crime2 <- crime2[crime2$state != "United States",]

pairs(crime2[,2:9], panel=panel.smooth)

#######################
# Plot 8 - Bubble Chart
#######################
crime <-
  read.csv("http://datasets.flowingdata.com/crimeRatesByState2005.tsv",
           header=TRUE, sep="\t")
radius <- sqrt(crime$population/ pi)
symbols(crime$murder, crime$burglary, circles=radius, inches=0.35
        ,fg="white", bg="red", xlab="Murder Rate", ylab="Burglary Rate")
text(crime$murder, crime$burglary, crime$state, cex=0.5)

####################
# Plot 9 - Histogram
####################
f <- file.choose()
birth <- read.csv(f
                  ,header=TRUE, sep=",")

hist(birth$X2021
     ,xlab = "Births per Woman"
     ,main = "GLOBAL DISTRIBUTION OF FERTILITY RATES")
View(birth)

########################
# Plot 10 - Density Plot
########################
birth2008 <- birth$X2008[!is.na(birth$X2008)]
d2008 <- density(birth2008)
#d2008frame <- data.frame(d2008$x, d2008$y)

plot(d2008, type='n')
polygon(d2008, col="#821122", border="#cccccc")

###########################
# Plot 11 - Wine Histograms
###########################
f <- file.choose()
wine <- read.csv(f,header=TRUE, sep=",")
View(wine)

par(mfrow=c(3,3))
plot.new()
hist(wine[wine$rep.region == "North",]$income)
plot.new()
hist(wine[wine$rep.region == "West",]$income)
hist(wine[wine$rep.region == "Central",]$income)
hist(wine[wine$rep.region == "East",]$income)
plot.new()
hist(wine[wine$rep.region == "South",]$income)

############################
# Plot 12 - Wine Time series
############################

wine_years <- tapply(wine$income, list(wine$year,wine$rep.region), sum)
wy <- as.data.frame(wine_years)
plot.new()
plot(rownames(wy),wy$North, type = 'l', ylim = c(0,max(wine_years))
     ,main = 'North')
plot.new()
plot(rownames(wy),wy$West, type = 'l', ylim = c(0,max(wine_years))
     ,main = 'West')
plot(rownames(wy),wy$Central, type = 'l', ylim = c(0,max(wine_years))
     ,main = 'Central')
plot(rownames(wy),wy$East, type = 'l', ylim = c(0,max(wine_years))
     ,main = 'East')
plot.new()
plot(rownames(wy),wy$South, type = 'l', ylim = c(0,max(wine_years))
     ,,main = 'South')

