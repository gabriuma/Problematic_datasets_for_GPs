library(rstan)

load("C:/GABRIEL_20180206/GIFLE/GIT/Problematic datasets for SE exact-GP/dataset2/f_true.rData")

load("C:/GABRIEL_20180206/GIFLE/GIT/Problematic datasets for SE exact-GP/dataset2/x.rData")

load("C:/GABRIEL_20180206/GIFLE/GIT/Problematic datasets for SE exact-GP/dataset2/y.rData")

# true_lscale= 0.10 (square exponential kernel)

n <- length(x)

standata_GP <- list(x= x, y= y, n= n)

stanout_GP <- stan(file= "C:/GABRIEL_20180206/GIFLE/GIT/Problematic datasets for SE exact-GP/dataset2/stancode_GP.stan", data= standata_GP, iter= 200,  warmup= 100, chains= 2, thin= 1, algorithm= "NUTS")


dev.new()
traceplot(stanout_GP, pars = c("lscale","sdgp","sigma"), include = TRUE, unconstrain = FALSE, inc_warmup = FALSE, window = NULL, nrow = NULL, ncol = NULL)


f <- summary(stanout_GP, pars = c("f"), probs = c(0.025, 0.5, 0.975), digits_summary = 4)$summary[,c(1,4,6)]


dev.new()
plot(x, y, type="p", cex=0.5, pch=4, col="grey")
lines(x, f_true, col="grey")
matplot(x, f, add=TRUE, type="l", col=2)
