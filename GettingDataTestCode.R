## Test 3
rm(list=ls())
library(dplyr)
gdp <- read.csv("gdp.csv", stringsAsFactors = FALSE, skip = 5, 
                header=FALSE, nrows=236, skipNul = TRUE, na.strings = "")
gdp<- select(gdp,V1,V2,V4,V5)
names(gdp)<-c("CountryCode","Rank","Long.Name","Value")
gdp$Rank <- as.numeric(gdp$Rank)
gdp<-filter(gdp, !is.na(Rank))
edu <- read.csv("edu.csv", stringsAsFactors = FALSE, na.strings = "")
edu<- select(edu,CountryCode, Long.Name, Income.Group, Special.Notes)

mrg <- merge(gdp, edu, by = "CountryCode")
mrgRank <- filter(mrg, !is.na(Rank))
mrgRank <- arrange(mrgRank, desc(Rank))
ctyx <- mrgRank$Long.Name.x[13]
ctyy <- mrgRank$Long.Name.y[13]
avgIn <- summarise(group_by(mrgRank,Income.Group), mean(Rank))
mrgQ <- quantile(mrgRank$Rank,seq(0,1,0.2))
gp1 <- filter(mrgRank, Rank >= mrgQ[1] & Rank <= mrgQ[2])
gp2 <- filter(mrgRank, Rank >= mrgQ[2] & Rank <= mrgQ[3])
gp3 <- filter(mrgRank, Rank >= mrgQ[3] & Rank <= mrgQ[4])
gp4 <- filter(mrgRank, Rank >= mrgQ[4] & Rank <= mrgQ[5])
gp5 <- filter(mrgRank, Rank >= mrgQ[5] & Rank <= mrgQ[6])
gp1 <- mutate(gp1, quant = 1)
gp2 <- mutate(gp2, quant = 2)
gp3 <- mutate(gp3, quant = 3)
gp4 <- mutate(gp4, quant = 4)
gp5 <- mutate(gp5, quant = 5)
mrgRank <- rbind(gp1,gp2,gp3,gp4,gp5)
with(mrgRank, table(quant, Income.Group))

############################
## Test 4
rm(list=ls())
library(dplyr)
## Q1
## fileUrl <- "https://d396qusza40orc.cloudfront.net/getdata%2Fdata%2Fss06hid.csv"
## download.file(fileUrl, "q1.csv")
## mydf <- read.csv("q1.csv", stringsAsFactor = FALSE)
## cnames <- names(mydf)
## mysp <- sapply(cnames, strsplit,"wgtp")
## print(mysp[123])

## Q2
## download.file("https://d396qusza40orc.cloudfront.net/getdata%2Fdata%2FGDP.csv", "q2.csv")
Q2 <- function(){
        mydf <- read.csv("q2.csv", stringsAsFactor = FALSE, header = FALSE,quote = "\"")
        mydf$V2 <- as.numeric(mydf$V2)
        mydf <- filter(mydf, !is.na(V2))
        sValue <- mydf$V5
        Value <- strsplit(sValue, ",", fixed = TRUE)
        nValue <- c(1:190)
        for(i in 1:190){
                tmp <- as.array(Value[[i]])
                tmp2 <- ifelse(is.na(tmp[3]),
                               ifelse(is.na(tmp[2]),
                                      tmp[1],
                                      paste(tmp[1],tmp[2],sep="")),
                              paste(tmp[1],tmp[2],tmp[3],sep=""))
                nValue[i] <- as.numeric(tmp2)
        }
        print(mean(nValue))
}

## Q3
Q3 <- function(){
        mydf <- read.csv("q2.csv", stringsAsFactor = FALSE, header = FALSE,quote = "\"")
        mydf$V2 <- as.numeric(mydf$V2)
        mydf <- filter(mydf, !is.na(V2))
        countryNames <- mydf$V4
        ctUnited <- length(grep("^United",countryNames))
        print(ctUnited)
}

## Q4
Q4 <- function(){
        gdp <- read.csv("gdp.csv", stringsAsFactors = FALSE, skip = 5, 
                        header=FALSE, nrows=236, skipNul = TRUE, na.strings = "")
        gdp<- select(gdp,V1,V2,V4,V5)
        names(gdp)<-c("CountryCode","Rank","Long.Name","Value")
        gdp$Rank <- as.numeric(gdp$Rank)
        gdp<-filter(gdp, !is.na(Rank))
        edu <- read.csv("edu.csv", stringsAsFactors = FALSE, na.strings = "")
        edu<- select(edu,CountryCode, Long.Name, Income.Group, Special.Notes)
        mrg <- merge(gdp, edu, by = "CountryCode")
        
        ##spc <- mrg$Special.Notes
        mrg1 <- mutate(mrg, fiscal = 0)
        fis <- grep("iscal year end", mrg$Special.Notes)
        for(i in fis){mrg1$fiscal[i] <- 1}
        mrg1 <- filter(mrg1, fiscal==1)
        fisjun <- length(grep("June", mrg1$Special.Notes))
        print(fisjun)
}

## Q5
Q5 <- function(){
        library(quantmod)
        amzn <- getSymbols("AMZN",auto.assign=FALSE)
        sampleTimes <- index(amzn)
        ftimes <- sapply(amzn, index)
        fdate <- ftimes[,1]
        a2012 <- which(fdate >= as.numeric(as.Date("2012-01-01")) & 
                        fdate <= as.numeric(as.Date("2012-12-31")))
        amar <- which(fdate >= as.numeric(as.Date("2012-03-01")) & 
                       fdate <= as.numeric(as.Date("2012-03-31")))
        print(length(a2012))
        print(length(amar))
}
