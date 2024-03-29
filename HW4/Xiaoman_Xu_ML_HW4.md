ML_HW4
================
Xiaoman Xu

``` r
library('MASS') ## for 'mcycle'
library('manipulate') ## for 'manipulate'
```

``` r
y <- mcycle$accel
x <- matrix(mcycle$times, length(mcycle$times), 1)

plot(x, y, xlab="Time (ms)", ylab="Acceleration (g)")
```

![](Xiaoman_Xu_ML_HW4_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

``` r
## Epanechnikov kernel function
## x  - n x p matrix of training inputs
## x0 - 1 x p input where to make prediction
## lambda - bandwidth (neighborhood size)
kernel_epanechnikov <- function(x, x0, lambda=1) {
  d <- function(t)
    ifelse(t <= 1, 3/4*(1-t^2), 0)
  z <- t(t(x) - x0)
  d(sqrt(rowSums(z*z))/lambda)
}

## k-NN kernel function
## x  - n x p matrix of training inputs
## x0 - 1 x p input where to make prediction
## k  - number of nearest neighbors
kernel_k_nearest_neighbors <- function(x, x0, k=1) {
  ## compute distance betwen each x and x0
  z <- t(t(x) - x0)
  d <- sqrt(rowSums(z*z))

  ## initialize kernel weights to zero
  w <- rep(0, length(d))
  
  ## set weight to 1 for k nearest neighbors
  w[order(d)[1:k]] <- 1
  
  return(w)
}

## Make predictions using the NW method
## y  - n x 1 vector of training outputs
## x  - n x p matrix of training inputs
## x0 - m x p matrix where to make predictions
## kern  - kernel function to use
## ... - arguments to pass to kernel function
nadaraya_watson <- function(y, x, x0, kern, ...) {
  k <- t(apply(x0, 1, function(x0_) {
    k_ <- kern(x, x0_, ...)
    k_/sum(k_)
  }))
  yhat <- drop(k %*% y)
  attr(yhat, 'k') <- k
  return(yhat)
}

## Helper function to view kernel (smoother) matrix
matrix_image <- function(x) {
  rot <- function(x) t(apply(x, 2, rev))
  cls <- rev(gray.colors(20, end=1))
  image(rot(x), col=cls, axes=FALSE)
  xlb <- pretty(1:ncol(x))
  xat <- (xlb-0.5)/ncol(x)
  ylb <- pretty(1:nrow(x))
  yat <- (ylb-0.5)/nrow(x)
  axis(3, at=xat, labels=xlb)
  axis(2, at=yat, labels=ylb)
  mtext('Rows', 2, 3)
  mtext('Columns', 3, 3)
}

## Compute effective df using NW method
## y  - n x 1 vector of training outputs
## x  - n x p matrix of training inputs
## kern  - kernel function to use
## ... - arguments to pass to kernel function
effective_df <- function(y, x, kern, ...) {
  y_hat <- nadaraya_watson(y, x, x,
    kern=kern, ...)
  sum(diag(attr(y_hat, 'k')))
}

## loss function
## y    - train/test y
## yhat - predictions at train/test x
loss_squared_error <- function(y, yhat)
  (y - yhat)^2

## test/train error
## y    - train/test y
## yhat - predictions at train/test x
## loss - loss function
error <- function(y, yhat, loss=loss_squared_error)
  mean(loss(y, yhat))

## AIC
## y    - training y
## yhat - predictions at training x
## d    - effective degrees of freedom
aic <- function(y, yhat, d)
  error(y, yhat) + 2/length(y)*d

## BIC
## y    - training y
## yhat - predictions at training x
## d    - effective degrees of freedom
bic <- function(y, yhat, d)
  error(y, yhat) + log(length(y))/length(y)*d


## make predictions using NW method at training inputs
y_hat <- nadaraya_watson(y, x, x,
  kernel_epanechnikov, lambda=5)

## view kernel (smoother) matrix
matrix_image(attr(y_hat, 'k'))
```

![](Xiaoman_Xu_ML_HW4_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
## compute effective degrees of freedom
edf <- effective_df(y, x, kernel_epanechnikov, lambda=5)
aic(y, y_hat, edf)
```

    ## [1] 677.1742

``` r
bic(y, y_hat, edf)
```

    ## [1] 677.3629

``` r
## create a grid of inputs 
x_plot <- matrix(seq(min(x),max(x),length.out=100),100,1)

## make predictions using NW method at each of grid points
y_hat_plot <- nadaraya_watson(y, x, x_plot,
  kernel_epanechnikov, lambda=1)

## plot predictions
plot(x, y, xlab="Time (ms)", ylab="Acceleration (g)")
lines(x_plot, y_hat_plot, col="#882255", lwd=2) 
```

![](Xiaoman_Xu_ML_HW4_files/figure-gfm/unnamed-chunk-3-2.png)<!-- -->

``` r
## how does k affect shape of predictor and eff. df using k-nn kernel ?
# manipulate({
#   ## make predictions using NW method at training inputs
#   y_hat <- nadaraya_watson(y, x, x,
#     kern=kernel_k_nearest_neighbors, k=k_slider)
#   edf <- effective_df(y, x, 
#     kern=kernel_k_nearest_neighbors, k=k_slider)
#   aic_ <- aic(y, y_hat, edf)
#   bic_ <- bic(y, y_hat, edf)
#   y_hat_plot <- nadaraya_watson(y, x, x_plot,
#     kern=kernel_k_nearest_neighbors, k=k_slider)
#   plot(x, y, xlab="Time (ms)", ylab="Acceleration (g)")
#   legend('topright', legend = c(
#     paste0('eff. df = ', round(edf,1)),
#     paste0('aic = ', round(aic_, 1)),
#     paste0('bic = ', round(bic_, 1))),
#     bty='n')
#   lines(x_plot, y_hat_plot, col="#882255", lwd=2) 
# }, k_slider=slider(1, 10, initial=3, step=1))
```

\#1. Randomly split the mcycle data into training (75%) and validation
(25%) subsets

``` r
train_idx <- sample(nrow(mcycle), round(0.75 * nrow(mcycle)))
train_data <- mcycle[train_idx, ]
validation_data <- mcycle[-train_idx, ]
```

\#2. Using the mcycle data, consider predicting the mean acceleration as
a function of time. Use the Nadaraya-Watson method with the k-NN kernel
function to create a series of prediction models by varying the tuning
parameter over a sequence of values. (hint: the script already
implements this)

``` r
## make predictions using NW method at training inputs
y_hat <- nadaraya_watson(y, x, x,
  kernel_epanechnikov, lambda=5)

## view kernel (smoother) matrix
matrix_image(attr(y_hat, 'k'))
```

![](Xiaoman_Xu_ML_HW4_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
## create a grid of inputs 
x_plot <- matrix(seq(min(x),max(x),length.out=100),100,1)

## make predictions using NW method at each of grid points
y_hat_plot <- nadaraya_watson(y, x, x_plot,
  kernel_epanechnikov, lambda=1)

## plot predictions
plot(x, y, xlab="Time (ms)", ylab="Acceleration (g)")
lines(x_plot, y_hat_plot, col="#882255", lwd=2) 
```

![](Xiaoman_Xu_ML_HW4_files/figure-gfm/unnamed-chunk-5-2.png)<!-- -->

\#3. With the squared-error loss function, compute and plot the training
error, AIC, BIC, and validation error (using the validation data) as
functions of the tuning parameter.

``` r
## compute effective degrees of freedom
edf <- effective_df(y, x, kernel_epanechnikov, lambda=5)
aic(y, y_hat, edf)
```

    ## [1] 677.1742

``` r
bic(y, y_hat, edf)
```

    ## [1] 677.3629

``` r
# manipulate({
# # make predictions using NW method at training inputs
#   y_hat <- nadaraya_watson(y, x, x,
#     kern=kernel_k_nearest_neighbors, k=k_slider)
#   edf <- effective_df(y, x,
#     kern=kernel_k_nearest_neighbors, k=k_slider)
#  # train_error <- error(y, yhat, lo)
#   aic_ <- aic(y, y_hat, edf)
#   bic_ <- bic(y, y_hat, edf)
#   y_hat_plot <- nadaraya_watson(y, x, x_plot,
#     kern=kernel_k_nearest_neighbors, k=k_slider)
#   plot(x, y, xlab="Time (ms)", ylab="Acceleration (g)")
#   legend('topright', legend = c(
#     paste0('eff. df = ', round(edf,1)),
#     paste0('aic = ', round(aic_, 1)),
#     paste0('bic = ', round(bic_, 1))),
#     bty='n')
#   lines(x_plot, y_hat_plot, col="#882255", lwd=2)
# }, k_slider=slider(1, 10, initial=3, step=1))
```

\#4. For each value of the tuning parameter, Perform 5-fold
cross-validation using the combined training and validation data. This
results in 5 estimates of test error per tuning parameter value.

``` r
library(boot)

tuning_params <- seq(0.1, 10, by=0.1)
test_errors <- matrix(NA, nrow=6, ncol=length(tuning_params)) ## change nrow to 6

for (i in 1:length(tuning_params)) {
  lambda <- tuning_params[i]
  
  ## combine training and validation data
  train_validation_data <- rbind(train_data, validation_data)
  
  ## define the glm model
  model <- glm(accel ~ times, data=train_validation_data)
  
  ## perform 5-fold cross-validation
  cv_results <- cv.glm(train_validation_data, model, K=5)
  
  ## extract the mean test error from the cv.glm output
  test_errors[,i] <- cv_results$delta[1] ## use only the mean test error
}

## take the average of the test errors across the 5 folds for each tuning parameter
mean_test_errors <- apply(test_errors, 2, mean)

mean_test_errors
```

    ##   [1] 2167.354 2130.328 2157.820 2140.674 2207.671 2147.061 2168.963 2173.054
    ##   [9] 2150.557 2161.934 2157.816 2185.553 2189.073 2178.381 2163.326 2160.519
    ##  [17] 2143.932 2177.807 2181.422 2140.622 2151.623 2240.288 2161.078 2234.482
    ##  [25] 2146.832 2168.767 2139.835 2180.084 2160.151 2143.871 2135.949 2168.437
    ##  [33] 2181.315 2174.490 2206.122 2200.398 2156.062 2248.514 2234.040 2149.543
    ##  [41] 2144.074 2162.025 2146.261 2144.819 2153.631 2155.548 2146.026 2136.511
    ##  [49] 2176.475 2191.724 2138.345 2166.288 2169.794 2191.232 2234.253 2188.340
    ##  [57] 2156.059 2193.369 2149.876 2181.220 2209.936 2149.734 2163.980 2160.936
    ##  [65] 2161.721 2154.302 2165.609 2247.004 2143.433 2136.011 2179.830 2174.499
    ##  [73] 2160.015 2188.564 2148.257 2125.505 2195.516 2157.634 2162.042 2181.922
    ##  [81] 2153.306 2163.392 2183.161 2132.894 2169.923 2136.853 2159.293 2157.071
    ##  [89] 2172.126 2157.929 2141.587 2202.821 2157.259 2232.624 2138.661 2296.790
    ##  [97] 2135.814 2314.799 2143.366 2221.341

\#5. Plot the CV-estimated test error (average of the five estimates
from each fold) as a function of the tuning parameter. Add vertical line
segments to the figure (using the segments function in R) that represent
one “standard error” of the CV-estimated test error (standard deviation
of the five estimates from each fold).

``` r
## calculate the standard errors of the test errors across the 5 folds for each tuning parameter
se_test_errors <- apply(test_errors, 2, sd) / sqrt(5)

## plot the mean test errors and standard errors as error bars
plot(tuning_params, mean_test_errors, type='l', xlab='Tuning parameter', ylab='Test error')
#axis(1, at=seq(0.1, 10, by=0.1))
segments(tuning_params, mean_test_errors - se_test_errors, tuning_params, mean_test_errors + se_test_errors, lwd=2)
abline(h=mean(mean_test_errors) + mean(se_test_errors), lty=1, col="red")
```

![](Xiaoman_Xu_ML_HW4_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

\#6. Interpret the resulting figures and select a suitable value for the
tuning parameter. - The resulting figure shows the testing error based
on the tuning parameter has a sequenced distribution from 0 to 10 by
0.1. On a moderate scale, the test error seems to decrease more as the
tuning parameter increases from 4.4. Based on the red abline, the
suitable value for the tuning parameter would be roughly around 5 \~ 6,
8 \~10. Specifically suitable values could be 5.6, 8.1, and 6.6
