# loading packages
library(spdep)
library(sf)
library(tidyverse)
library(dbscan)

#setwd(...) #set working directory if needed

# reading maps in sf format
# unzip "wojewozdzwa.zip" first, before running this code
WOJ<-st_read("wojewodztwa.shp")
WOJ<-st_transform(WOJ, 4326) 	# conversion to WGS84_
WOJ.maz<-WOJ[WOJ$jpt_nazwa=="mazowieckie", ]

#######################
# point data about firms and population - basic
daneOK<-read.table("daneOK9.txt")
popul<-read.csv("points_popul_maz.csv", header=TRUE, dec=".", sep=",")

# data in sf - firms
daneOK.sf<-st_as_sf(daneOK, coords = c("lon","lat"), crs = "+proj=longlat +datum=NAD83")


# data in sf - population
popul.sf<-st_as_sf(popul, coords = c("x","y"), crs = "+proj=longlat +datum=NAD83")


### generating summarising variables and standardization ###############

# aglomeration sectors
daneOK$locAggAgri<-daneOK$locAggA 	 # sec A

daneOK$locAggProd<-daneOK$locAggB + daneOK$locAggC + daneOK$locAggD + daneOK$locAggE	# sec BCDE

daneOK$locAggConstr<-daneOK$locAggF		# sec F

daneOK$locAggServ<-daneOK$locAggG	+ daneOK$locAggH + daneOK$locAggI + daneOK$locAggJ + daneOK$locAggK + daneOK$locAggL + daneOK$locAggM + daneOK$locAggN + daneOK$locAggO + daneOK$locAggP + daneOK$locAggQ + daneOK$locAggR + daneOK$locAggS + daneOK$locAggT + daneOK$locAggU # sec G-U

# binary dummies about sectors
daneOK$dummy_agri<-ifelse(daneOK$SEK_PKD7=="A",1,0)

daneOK$dummy_prod<-ifelse(daneOK$SEK_PKD7=="B" | daneOK$SEK_PKD7=="C" | daneOK$SEK_PKD7=="D" | daneOK$SEK_PKD7=="E",1,0)

daneOK$dummy_constr<-ifelse(daneOK$SEK_PKD7=="F",1,0)

daneOK$dummy_serv<-ifelse(daneOK$SEK_PKD7=="G" | daneOK$SEK_PKD7=="H" | daneOK$SEK_PKD7=="I" | daneOK$SEK_PKD7=="J" | daneOK$SEK_PKD7=="K" | daneOK$SEK_PKD7=="L" | daneOK$SEK_PKD7=="M" | daneOK$SEK_PKD7=="N" | daneOK$SEK_PKD7=="O" | daneOK$SEK_PKD7=="P" | daneOK$SEK_PKD7=="Q" | daneOK$SEK_PKD7=="R" | daneOK$SEK_PKD7=="S" | daneOK$SEK_PKD7=="T" | daneOK$SEK_PKD7=="U",1,0)

# standardisation of variables
daneOK$locPdens.s<-as.numeric(scale(daneOK$locPdens))
daneOK$locAggAgri.s<-as.numeric(scale(daneOK$locAggAgri))
daneOK$locAggProd.s<-as.numeric(scale(daneOK$locAggProd))
daneOK$locAggConstr.s<-as.numeric(scale(daneOK$locAggConstr))
daneOK$locAggServ.s<-as.numeric(scale(daneOK$locAggServ))
daneOK$locHH.s<-as.numeric(scale(daneOK$locHH))
daneOK$locHightech.s<-as.numeric(scale(daneOK$locHightech))
daneOK$locBIG.s<-as.numeric(scale(daneOK$locBIG))
daneOK$locAggTOTAL.s<-as.numeric(scale(daneOK$locAggTOTAL))
daneOK$locLQ.s<-as.numeric(scale(daneOK$locLQ))


# other variables
daneOK$dummy_ifbig<-ifelse(daneOK$GR_LPRAC>3,1,0)

# logarithms
daneOK$locAggTOTAL[daneOK$locAggTOTAL<2]<-2
daneOK$locPdens[daneOK$locPdens<2]<-2

daneOK$log_locAggTOTAL<-log(daneOK$locAggTOTAL)
daneOK$log_locPdens<-log(daneOK$locPdens)


daneOK <- daneOK %>% mutate(class4 = case_when(dummy_agri==1 ~ "agri",
                                               dummy_prod==1 ~ "prod",
                                               dummy_constr==1 ~ "constr",
                                               dummy_serv==1 ~"serv",
                                               TRUE ~ "serv"
)) %>%
  mutate(class4 = factor(class4))


#### Dane OK in sf with all variables #####################################
daneOK.full.sf<-st_as_sf(daneOK, coords = c("lon","lat"), crs = "+proj=longlat +datum=NAD83")
