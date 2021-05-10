write.csv(verlander, "C:\\Users\\Will\\Documents\\Verlander.csv", row.names = FALSE)
F4verl <- subset(verlander, pitch_type == "FF")
F4verl$gameDay <- as.integer(format(F4verl$gamedate, format="%j"))
dailySpeed <- aggregate(speed ~ gameDay + season, data=F4verl,FUN=mean)
avgSpeed <- aggregate(speed ~ pitches + season, data=F4verl,
                      FUN=mean)
xyplot(speed ~ pitches | factor(season),
       data=avgSpeed)
avgSpeedComb <- mean(F4verl$speed)


NoHit <- subset(verlander, gamedate == "2011-05-07")



write.csv(cabrera, "C:\\Users\\Will\\Documents\\Cabrera.csv", row.names = FALSE)


bases <- data.frame(x=c(0, 90/sqrt(2), 0, -90/sqrt(2), 0),y=c(0, 90/sqrt(2), 2 * 90/sqrt(2), 90/sqrt(2), 0))


library(dplyr)
#change NA's to 0s
craig_data <- craig_data %>%
  mutate(persons_vaccinated_unique_dose = coalesce(persons_vaccinated_unique_dose, 0),
         persons_vaccinated_one_plus_dose = coalesce(persons_vaccinated_one_plus_dose, 0),
         doses_administered = coalesce(doses_administered,0))
#create a subset of the countries that need to be combined
countries_to_combine = filter(craig_data, country_code == 'BON'| country_code =='SAB')
#name the country and country code that should be kept, and combine data based on as_of_date
df1 = countries_to_combine %>% 
  group_by(country_name = "Bonaire", country_code = "BON", as_of_date)%>%
    summarise(persons_vaccinated_unique_dose = sum(persons_vaccinated_unique_dose), 
              persons_vaccinated_one_plus_dose = sum(persons_vaccinated_one_plus_dose),
              doses_administered = sum(doses_administered),
              persons_vaccinated_full= sum(persons_vaccinated_full))
#remove these combined countries from the original data set
craig_data = filter(craig_data, country_code != 'BON' &  country_code != 'SAB')
#add the combined countries data to the original data set
craig_data = rbind(df1,craig_data)



