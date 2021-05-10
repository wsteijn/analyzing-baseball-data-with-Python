data2011 <- read.csv("all2011.csv", header=FALSE)
fields <- read.csv("fields.csv")
names(data2011) <- fields[, "Header"]
data2011$RUNS <- with(data2011, AWAY_SCORE_CT + HOME_SCORE_CT)
data2011$HALF.INNING <- with(data2011, paste(GAME_ID, INN_CT, BAT_HOME_ID))

data2011$RUNS.SCORED <- with(data2011, (BAT_DEST_ID > 3) + (RUN1_DEST_ID > 3) + (RUN2_DEST_ID > 3) + (RUN3_DEST_ID > 3))
RUNS.SCORED.INNING <- aggregate(data2011$RUNS.SCORED,list(HALF.INNING=data2011$HALF.INNING), sum)
RUNS.SCORED.START <- aggregate(data2011$RUNS,list(HALF.INNING=data2011$HALF.INNING), "[", 1)

MAX <- data.frame(HALF.INNING=RUNS.SCORED.START$HALF.INNING)
MAX$x <- RUNS.SCORED.INNING$x + RUNS.SCORED.START$x
data2011 <- merge(data2011, MAX)
N <- ncol(data2011)
names(data2011)[N] <- "MAX.RUNS"
data2011$RUNS.ROI <- with(data2011, MAX.RUNS - RUNS)

table(data2011$RUNS.ROI)



RUNNER1 <- ifelse(as.character(data2011[ ,"BASE1_RUN_ID"]) == "", 0, 1)
RUNNER2 <- ifelse(as.character(data2011[ ,"BASE2_RUN_ID"]) == "", 0, 1)
RUNNER3 <- ifelse(as.character(data2011[ ,"BASE3_RUN_ID"]) == "", 0, 1)
RUNNER1[12443]

get.state <- function(runner1, runner2, runner3, outs){
  runners <- paste(runner1, runner2, runner3, sep="")
  paste(runners, outs)
}
data2011$STATE <- get.state(RUNNER1, RUNNER2, RUNNER3, data2011$OUTS_CT)


NRUNNER1 <- with(data2011, as.numeric(RUN1_DEST_ID == 1 |BAT_DEST_ID == 1))
NRUNNER2 <- with(data2011, as.numeric(RUN1_DEST_ID == 2 |RUN2_DEST_ID == 2 | BAT_DEST_ID==2))
NRUNNER3 <- with(data2011, as.numeric(RUN1_DEST_ID == 3 |RUN2_DEST_ID == 3 | RUN3_DEST_ID == 3 | BAT_DEST_ID == 3))
NRUNNER1
NOUTS <- with(data2011, OUTS_CT + EVENT_OUTS_CT)
data2011$NEW.STATE <- get.state(NRUNNER1, NRUNNER2, NRUNNER3, NOUTS)
data2011 <- subset(data2011, (STATE != NEW.STATE) | (RUNS.SCORED > 0))

dim(table(data2011$STATE))
dim(table(data2011$NEW.STATE))
table(NRUNNER1)





library(plyr)
data.outs <- ddply(data2011, .(HALF.INNING), summarize,
                   Outs.Inning=sum(EVENT_OUTS_CT))
data2011 <- merge(data2011, data.outs)
data2011C <- subset(data2011, Outs.Inning == 3)
data.outs



RUNS <- with(data2011C, aggregate(RUNS.ROI, list(STATE), mean))
RUNS$Outs <- substr(RUNS$Group, 5, 5)
RUNS <- RUNS[order(RUNS$Outs), ]


RUNS.out <-matrix(round(RUNS$x, 2), 8, 3)
dimnames(RUNS.out)[[2]] <- c("0 outs", "1 out", "2 outs")
dimnames(RUNS.out)[[1]] <- c("000", "001", "010", "011", "100", "101",
                             "110", "111")

RUNS.2002 <- matrix(c(.51, 1.40, 1.14, 1.96, .90, 1.84, 1.51, 2.33,
                      .27, .94, .68, 1.36, .54, 1.18, .94, 1.51,
                      .10, .36, .32, .63, .23, .52, .45, .78), 8, 3)
dimnames(RUNS.2002) <- dimnames(RUNS.out)

cbind(RUNS.out, RUNS.2002)


RUNS.POTENTIAL <- matrix(c(RUNS$x, rep(0, 8)), 32, 1)
dimnames(RUNS.POTENTIAL)[[1]] <- c(RUNS$Group, "000 3", "001 3",
                                   "010 3", "011 3", "100 3", "101 3", "110 3", "111 3")
data2011$RUNS.STATE <- RUNS.POTENTIAL[data2011$STATE, ]
data2011$RUNS.NEW.STATE <- RUNS.POTENTIAL[data2011$NEW.STATE, ]
data2011$RUNS.VALUE <- data2011$RUNS.NEW.STATE - data2011$RUNS.STATE +
  data2011$RUNS.SCORED

RUNS.POTENTIAL["000 3",]


table(data2011$RUNS.STATE)

















