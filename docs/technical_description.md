# Introduction

Forecasters are commonly asked to account for *known future events*. These are large business events which will have an effect on the forecast for e.g. the sales of a product, which it is known will happen ahead of time. Examples could be:

* An existing product entering a new market
* A price change to a product
* ...

The business will often have critical information about how they believe these events will affect the sales of a given product. At the time, this information is the best we have as to how this event will affect product sales. Since they will cause significant bias in an actuals only based forecasting algorithm, it is important to incorporate this information into the forecast.

One option is to use *judgmental post-adjustment*. This is where business experts will change the output of a statistical forecast based on what they think will happen when this event occurs. This typically works quite well *before* an event, as it is adjusting the forecast for information it is unable to use. But after an event has occurred, it's difficult to know how much information about the event the statistical forecast is using. Since the outcome of these events can vary rapidly, and the outcome can be very different to what is expected, business information can become outdated. This can lead to adjusted forecasts that are more inaccurate than the statistical base.

Another option is to use regressors. We tried this method ourselves. It worked well before the event as it would more often than not adjust the forecasting algorithm for its known bias. Once again though, after the event occurred this regressor degraded quickly even if the outcome of the event deviated only slightly from what was expected (which happens most of the time). This could lead to large forecasting errors.

`judgyprophet` aims to solve this problem by encoding these events using Bayesian informative priors. Before the event occurs, `judgyprophet` uses the business information to adjust likely biases in its statistical forecast due to a known business event. After the event, `judgyprophet` adjusts its estimate using Bayesian updating. This attempts to balance the expert knowledge of the business, with what is actually being observed in the actuals. This enables it to adjust rapidly as the event deviates from what is expected by the business.

As the name suggests, `judgyprophet` is based on the `prophet` algorithm, and uses concept of changepoints. It splits the changepoints into two groups: business known changepoints -- which are used to encode events with known business information; business unspecified changepoints -- which act more like the `prophet` changepoints and enable the forecast to adjust to unanticipated changes in trend and level of the timeseries.


# The forecasting model

Similar to Prophet, we use a decomposable timeseries model
\\[
    y(t) = g(t) + s(t) + \epsilon_t,
\\]
where \\(g(t)\\) is the trend function, and \\(s(t)\\) is the seasonality function.

The main difference being we further decompose \\(g(t)\\) into three components
\\[
    g(t) = L(t) + M(t) + U(t).
\\]
These components are defined as follows:

* \\(L(t)\\) -- business known level events. These are events that lead to sudden level shifts in the timeseries that the business know about ahead of time. An example might be the \$ sales for an inelastic good (e.g. food) when there is a price increase.
* \\(M(t)\\) -- business known trend events. These are events that lead to a steady increase in the target that the business know about ahead of time. An example might be the product sales for an existing product that is entering a new market (e.g. a console game being released in a new country).
* \\(U(t)\\) -- unspecified changepoint events. These are events that cause a change in trend that are unanticipated by the business ahead of time. These are equivalent to Prophet's linear changepoints. They help to capture unanticipated changes in the target.

## Business Known Level Events: \\(L(t)\\)

Level events are business known events that will lead to sudden level shifts in the timeseries (see the quickstart tutorial for an example).

Suppose the business identify \\(J\\) possible level events, define the business estimate of the change in level for event \\(j \in J\\), which occurs at time \\(\tau_j\\), to be \\(l_{0j}\\), and define the parameter defining the unknown actual level change to be \\(l_j\\).

Then the level event model is defined as follows
\\[
    L(t) = \sum_{j=1}^J l_j \mathbf 1 \\{t \geq \tau_j \\},
\\]
where \\(c_l\\) is drawn from the following prior
\\[
    c_j \sim N(l_{0j}, \sigma_l^2), \qquad \text{for } j = 1, ..., J.
\\]
Here \\(N\\) denotes the normal distribution; \\(\sigma_c\\) is the standard deviation set for the level events, which is a settable parameter in `judgyprophet`.

The idea is that before the event is observed, \\(l_{0j}\\) is used as the estimate for the level change, as that is the best estimate we have. But as we move to observing the event, the actuals are used to learn a better value for \\(l_j\\) using Bayesian updating.

Because of the prior we have set, \\(l_j\\) will be penalized from taking values away from \\(l_{0j}\\). This ensures that \\(l_j\\) won't make large jumps from the business estimate when not much data has been observed. This stops the algorithm from making erratic forecasts. This technique is similar to the idea of ridge regression.


## Business Known Trend Events: \\(M(t)\\)


Trend events are business known events that will lead to a trend change in the timeseries after the event (e.g. a gradual and repeating increase in sales. See the quickstart tutorial for an example).

We start by discussing the simplest trend event available in `judgyprophet`: a linear one. It's important to think about damping in our model though. Often when a product is released to a new market, there is an increase in sales, followed by a steady slow down of this as the market is saturated. This can be modelled in `judgyprophet`, and is covered in the next subsection.

Suppose the business identify \\(I\\) possible trend events, define the business estimate of the change in trend for event \\(i\\), which occurs at time \\(\tau_i\\), to be \\(m_{0i}\\), and define the parameter defining the unknown actual trend change to be \\(m_i\\).

Then the trend event model is defined as follows

\\[
    M(t) = \sum_{i=1}^I m_i s_i(t).
\\]
Here we define \\(s_i(t) = \max\\{t - \tau_i, 0\\}\\), i.e.\ before the event time \\(\tau_i\\) \\(s_i(t)\\) is 0, and after that it measures the time since the event.

Similarly to for level events we define the prior for \\(m_i\\) to be
\\[
    m_i \sim N(m_{0i}, \sigma_m^2),
\\]
where \\(\sigma_m\\) is the standard deviation set for trend events. It sets the sensitivity of the model to the actuals post any trend event.

## Damping

We found using logistics curves, as in Prophet, to handle damping to be too sensitive to the maximum capacity constant (defined by \\(C\\) in Prophet). Therefore we instead use a model inspired by Holt's damped trend exponential smoothing model (see e.g. [Forecasting Principle and Practice. Hyndman, Athanasopoulos.](https://otexts.com/fpp3/holt.html)).

This model is very similar to the model used for linear trends. But we add one further hyperparameter for each event \\(\gamma_i\\). This parameter defines the level of damping for event \\(i\\) in exactly the same way as Holt's damped trend method (see e.g. [Forecasting Principle and Practice. Hyndman, Athanasopoulos.](https://otexts.com/fpp3/holt.html)). This constant is typically set between 0.8 and 1, where 1 is equivalent to a linear trend, and 0.8 indicates quite heavy damping.

First set \\(d(t)\\) to be the following
\\[
    d_i(t) = \frac{(1 - \gamma_i^t)}{1 - \gamma_i} \mathbf 1 \\{t > 0\\}.
\\]
Then we define the damped trend to be
\\[
    M(t) = \sum_{i=1}^I m_i d_i(\max\\{t - \tau_i, 0\\}).
\\]
We assign the same prior to \\(m_i\\) as in the linear trend case.

This is similar to Holt's damping with trend because we can expand \\(d_i(t)\\) using the properties of geometric series to get
\\[
    d_i(t) = \frac{1 - \gamma_i^t}{1 - \gamma_i} \mathbf 1 \\{t > 0\\} = \mathbf 1 \\{t > 0\\} \sum_{t=0}^{t-1} \gamma^t.
\\]
This is very similar in form to the exponential smoothing equations.

For now, \\(\gamma_i\\) is a hyperparameter rather than a learned Bayesian parameter, i.e.\ it will not be updated as we observe actuals. We found the parameter was quite poorly predicted when it is modelled. We have added trying to add this as a learnt parameter again to the roadmap.


## Unspecified changepoint events: \\(U(t)\\)

These events are very similar to Prophet's original concept of changepoints. These are sudden changes in level or trend in the training period of the timeseries. Unlike Prophet, we do not use these changes for prediction, only to capture observed, unanticipated events in the timeseries. They are normally evenly dispersed across the timeseries, or exact locations can be specified by the user.

Let there be \\(k \in K\\) unspecified changepoint events, occurring at time \\(\tau_k\\), let the unknown level and trend changes be \\(v_k\\) and \\(w_k\\) respectively, then we define
\\[
    U(t) = \sum_{k=1}^K v_k \mathbf 1\\{t \geq \tau_k\\} + w_k \max\\{t - \tau_k, 0\\}.
\\]
Similar to Prophet, in order to penalise using these unspecified changepoints when unnecessary, rather than a normal prior as for the above events, we use a Laplace prior. This is equivalent to setting an \\(L_1\\) or Lasso penalty on the changes -- hopefully sending them to 0 if they are unnecessary. Specifically we set
\\[
    v_k, w_k \sim \text{Laplace}(0, \sigma_U), \qquad \text{for }k = 1, ..., K.
\\]

## Seasonality


Similar to Prophet, we encode seasonality using Fourier series. The user defines the seasonal period \\(s_S\\), and the fourier period \\(s_F\\) (this is automatically set to \\(s_S - 1\\) if unspecified). Then seasonality is defined as
\\[
    s(t) = \sum_{j=1}^{s_F} \beta_j f_j(t),
\\]
where
\\[
    f_j(t) = \mathbf 1\\{j \text{mod} 2 = 0\\} \cos\left(\frac{2 \pi \lfloor j / 2 \rfloor t}{s_F}\right) + \mathbf 1\\{j \text{mod} 2 = 1\\} \sin\left(\frac{2 \pi (\lfloor j / 2 \rfloor + 1) t}{s_F}\right).
\\]
For each seasonal coefficient \\(\beta_j\\), we set the prior
\\[
    \beta_j \sim N(0, \sigma_s).
\\]
