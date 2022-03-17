functions {

  vector level_event(
    real c,
    vector delta_level,
    int T
  ) {
    vector[T] curve;

    curve = c * delta_level;
    return curve;
  }

  vector trend_event(
    real m,
    int is_damping,
    real gamma,
    vector delta_trend,
    vector indication_t,
    int T
  ) {
    vector[T] curve;

    if (is_damping) {
      for (s in 1:T) {
        curve[s] = (m * delta_trend[s]) .* (1 - gamma ^ indication_t[s]) / (1 - gamma);
      }
    } else {
      curve = (m * delta_trend) .* indication_t;
    }
    return curve;
  }

  vector changepoint_event (
    real cp_c,
    real cp_m,
    vector delta_cp,
    vector cp_t,
    int T
  ) {
    vector[T] curve;

    curve = delta_cp .* (cp_c + cp_m * cp_t);
    return curve;
  }

  vector get_seasonal_levels (
    int is_seasonal,
    int is_multiplicative_seasonality,
    int fourier_order,
    vector[] fourier_series,
    real[] fourier_coefficients,
    int T
  ) {
    vector[T] seasonal_level;

    if (is_multiplicative_seasonality) {
      seasonal_level = rep_vector(1, T);
    } else {
      seasonal_level = rep_vector(0, T);
    }

    if (is_seasonal) {
      for (j in 1:fourier_order) {
        seasonal_level += fourier_coefficients[j] * fourier_series[j];
      }
    }

    return seasonal_level;
  }
}


data {
  int T;                                      // Number of time periods
  vector[T] y;                                // Time series
  vector[T] t;                                // Vector of time points. Starting at 0.
  // Level event parameters
  int<lower=0> L;                             // Number of level events
  real c0[L];                                 // Initial level event size
  vector[L ? T: 0] delta_level[L ? L: 1];     // Vector of size T, delta_level[t] is 0 if level event
                                              // has not happened yet. Otherwise 1.
                                              // If there are no level events, set dims to (1,0)
                                              // (i.e. input will be a list of empty list)
  // Trend event parameters
  int<lower=0> M;                             // Number of trend events
  real m0[M];                                 // Initial gradient parameters
  int<lower=0, upper=1> is_damping[M];        // 0 if there is no damping, 1 otherwise
  real gamma[M];                              // Damping parameters
  vector[M ? T: 0] delta_trend[M ? M: 1];     // Vector of size T, delta_level[t] is 0 if trend event
                                              // has not happened yet. Otherwise 1.
                                              // If there are no trend events, set dims to (1,0)
                                              // (i.e. input will be a list of empty list)
  vector[M ? T: 0] trend_t[M ? M: 1];         // Vector of time points for each trend event.
                                              // Counts number of indexes since event start.
                                              // If there are no level events, set dims to (1,0)
                                              // (i.e. input will be a list of empty list)
  // Unspecified changepoint parameters
  int<lower=0> C;                             // Number of changepoint events
  vector[C ? T: 0] delta_cp[C ? C: 1];        // Vector of size T, delta_cp[t] is 0 if changepoint event
                                              // has not happened yet. Otherwise 1.
                                              // If there are no level events, set dims to (1,0)
                                              // (i.e. input will be a list of empty list)
  vector[C ? T: 0] cp_t[C ? C: 1];            // Vector of time points for each changepoint event.
                                              // Counts number of indexes since changepoint start.
                                              // If there are no level events, set dims to (1,0)
                                              // (i.e. input will be a list of empty list)
  // Seasonality
  int<lower=0, upper=1> is_seasonal;          // Determines whether seasonality should be applied
  int<lower=1> seasonal_period;               // Seasonal period
  int<lower=0> fourier_order;                 // Order of fourier component
  vector[is_seasonal ? T: 0] fourier_series[is_seasonal ? fourier_order: 1];
                                              // Fourier series coefficients
  int<lower=0, upper=1> is_multiplicative_seasonality;  // Additive or multiplicative seasonality.
  // kwargs
  real<lower=0> sigma_cp;                     // Hyper-standard-deviation of unspecified changepoint parameters.
                                              // This determines how much shrinkage there will be.
  real<lower=0> sigma_beta_0;                 // Hyper-standard-deviation of beta_0 (base bias)
  real<lower=0> sigma_beta_1;                 // Hyper-standard-deviation of beta_1 (base intercept)
  real<lower=0> sigma_level;                  // Hyper-standard-deviation of level events
  real<lower=0> sigma_trend;                  // Hyper-standard-deviation of trend events
  real<lower=0> sigma_seasonal;               // Hyper-standard-deviation of seasonal dummy variables
  real<lower=0> sigma_actuals;                // The sigma to use for the noise variable
}


parameters {
  real beta_0;                                              // Bias term
  real beta_1;                                              // Base trend
  real c[L];                                                // Learnt level event gradients
  real m[M];                                                // Learnt trend event gradients
  real cp_c[C];                                             // Learnt unspecified changepoint biases
  real cp_m[C];                                             // Learnt unspecified changepoint gradients
  real fourier_coefficients[fourier_order];                 // Seasonal dummy coefficients
}


transformed parameters {
  vector[T] trend;
  vector[T] seasonal_level;
  vector[T] seasonal_trend;

  // Base trend
  trend = beta_0 + beta_1 * t;
  // Level events
  if (L > 0) {
    for (i in 1:L) {
      trend += level_event(c[i], delta_level[i], T);
    }
  }
  // Trend events
  if (M > 0) {
    for (i in 1:M) {
      trend += trend_event(m[i], is_damping[i], gamma[i], delta_trend[i], trend_t[i], T);
    }
  }
  // Unspecified changepoint events
  if (C > 0) {
    for (i in 1:C) {
      trend += changepoint_event(cp_c[i], cp_m[i], delta_cp[i], cp_t[i], T);
    }
  }


  // Seasonal level using fourier series
  seasonal_level = get_seasonal_levels(
    is_seasonal,
    is_multiplicative_seasonality,
    fourier_order,
    fourier_series,
    fourier_coefficients,
    T
  );

  // Combine together trend and seasonal level to get 'seasonal trend'
  if (is_multiplicative_seasonality) {
    seasonal_trend = trend .* seasonal_level;
  } else {
    seasonal_trend = trend + seasonal_level;
  }
}

model {
  // Regression priors
  beta_0 ~ normal(0, sigma_beta_0);
  beta_1 ~ normal(0, sigma_beta_1);

  // Level event
  if (L > 0) {
    for (i in 1:L) {
      // Penalised prior to try to discourage overupdating early on after event
      c[i] ~ normal(c0[i], sigma_level);
    }
  }
  // Trend event
  if (M > 0) {
    for (i in 1:M) {
      // Penalised prior to try to discourage overupdating early on after event
      m[i] ~ normal(m0[i], sigma_trend);
    }
  }
  // Unspecified changepoint event
  if (C > 0) {
    for (i in 1:C) {
      // Penalised prior to try to discourage overupdating early on after event
      cp_c[i] ~ double_exponential(0, sigma_cp);
      cp_m[i] ~ double_exponential(0, sigma_cp);
    }
  }
  // Seasonality
  if (is_seasonal) {
    fourier_coefficients ~ normal(0, sigma_seasonal);
  }

  // Likelihood
  y ~ normal(seasonal_trend, sigma_actuals);
}
