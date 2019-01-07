#First time coding R

#Checking version
R.Version()

#Calculations
2*3
3^4
(50 + 7)/(8 * (3 - 5/2))
7 * 9 + 2 * 6
2.5 * 4

x <-15
x + 5
x * x / 2
2 ^ x
y <- x / 3
sqrt(16)
round(5.3499999, 2)

#Commands

install.packages("tidyverse")

?rm
?rnorm

#Importing Dataframe and checking it

library(tidyverse)
senado <- read_csv("senado.csv")
head(senado)
tail(senado)
class(senado)
str(senado)
summary(senado)

# Reading file with delimiter
read_delim('D:/DataScience/Courses/Data Science Brazil Marathon/Week_04/arquivo_separado_por#.txt', delim = '#')
#Reading CSV file with delimiter | and storing it in a variable
ta_precos_medicamentos <- read_delim("D:/DataScience/Courses/Data Science Brazil Marathon/Week_04/TA_PRECOS_MEDICAMENTOS.csv", delim = |)

#Reading TXT file with fixed widths columns
fwf_samples <- read_fwf("D:/DataScience/Courses/Data Science Brazil Marathon/Week_04/fwf-sample.txt", col_positions =
                          fwf_widths(c(20, 10, 12), c("nomes", "estado", "codigo")))

#Class Types
inteiro <- 928
outro.inteiro <- 5e2
decimal <- 182.93
caracter <- 'exportação'
logico <- TRUE
outro.logico <- FALSE
class(inteiro)
class(outro.inteiro)
class(decimal)
class(caracter)
class(logico)
class(outro.logico)

#Dataframes
class(senado)
dim(senado)

#Vectors
vetor.chr <- c('tipo1', 'tipo2', 'tipo3', 'tipo4')
vetor.num <- c(1, 2, 5, 8, 1001)
vetor.num.repetidos <- c(rep(2, 50)) #usando funcãopararepetirnúmeros
vetor.num.sequencia <- c(seq(from=0, to=100, by=5)) #usando funçãoparacriarsequências
vetor.logical <- c(TRUE, TRUE, TRUE, FALSE, FALSE)
vetor.chr
vetor.num
vetor.num.repetidos
vetor.num.sequencia
vetor.logical

#More tests
7 + TRUE
2015 > "2016"
"2014" < 2017
6 > "100"
1 + "1"
"6" < 5

#Testing while
automatico <- list.files('automatico/')
while (length(automatico) == 0) {
automatico <- list.files('D:/DataScience/Courses/Data Science Brazil Marathon/Week_04/automatico/')
if(length(automatico) > 0) {
print('O arquivochegou!')
print('Inicia aleituradosdados')
print('Faz amanipulação')
print('Envia email informando conclusao dos calculos')
} else {
print('aguardando arquivo...')
Sys.sleep(5)
}
}


senado2 <- senado %>%
  select(VoteNumber, SenNumber, SenatorUpper, Vote, Party, GovCoalition, State, FP, Origin, Contentious, PercentYes,
       IndGov, VoteType, Content, Round) %>%
  filter(is.na(State) == FALSE)
is.na(senado2$State)

senado2 %>%
  group_by(Party) %>%
  filter(GovCoalition == TRUE)


number_of_sen <- senado2 %>%
group_by(Party) %>%
summarise(qty_senators = n_distinct(SenNumber, na.rm=TRUE))
number_of_sen


number_of_yes <- senado2 %>%
group_by(Party) %>%
summarise(qty_yes = sum(Vote == 'S'))
number_of_yes


senado2 <- mutate(senado2, Regiao = ifelse(State%in%c('AM','AC','TO','PA','RO','RR'),'Norte',
                                           ifelse(State %in%c('SP','MG','RJ','ES'),'Sudeste',
                                                  ifelse(State%in%c('MT','MS', 'GO', 'DF'), 'Centro-Oeste',
                                                         ifelse(State%in%c('PR','SC','RS'),'Sul','Nordeste')))))














