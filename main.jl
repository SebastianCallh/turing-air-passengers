using Turing, Distributions, Plots, StatsPlots, DataFrames, CSV, Dates

function save_fig(fig, filename)
    savefig(fig, "plots/$filename")    
end

# Data loading 
df = CSV.read("data/AirPassengers.csv", DataFrame) |>
    df -> rename(df, ["Month", "Passengers"])

@df df scatter(
    :Month, :Passengers,
    label=nothing,
    xlabel="Month",
    ylabel="Number of passengers",
    title="Airplane passengers data"
)

## Standardise and plot 
function prepare_t(months, maxmonth=nothing)
    t = months
    t = month.(t) .+ 12*year.(t)
    tmin = minimum(t)
    tmax = maxmonth !== nothing ? maxmonth : maximum(t)
    t = (t .- tmin) ./ (tmax - tmin)
end

y = Float32.(df.Passengers)
ymax = maximum(y)
y = y ./ ymax
t = prepare_t(df.Month)

scatter(
    t, y,
    label=nothing,
    title="Normalised data",
    xlabel="x",
    ylabel="Number of passengers"
)

# Simple linear model
@model function linear(t, y)
    α ~ Normal(0, .5)
    β ~ Normal(0, .5)
    σ ~ TruncatedNormal(0, .5, 0, Inf)
    trend = @. α + β * t
    y ~ MvNormal(trend, σ)
    (; trend)
end

function plot_samples(t, samples, ymax; args...)
    y_samples = group(samples, :y).value[:,:,1]
    plt = plot(t ,transpose(y_samples)*ymax, label=nothing, color=1, alpha=0.2)
    plot!(plt, [], [], color=1; args...)
    plt
end

## Prior predictive
linear_prior_samples = sample(linear(t, missing), Prior(), 100)
linear_prior_plt = plot_samples(df.Month, linear_prior_samples, ymax, label="Data", title="Prior samples")
@df df scatter!(linear_prior_plt, :Month, :Passengers, label="Data", color=2, legend=:topleft)
save_fig(linear_prior_plt, "linear_prior_predictive.png")

## Posterior predictive
linear_posterior_chain = sample(linear(t, y), NUTS(), 1000)
linear_posterior_samples = predict(linear(t, missing), linear_posterior_chain)
linear_posterior_plt = plot_samples(
    df.Month, linear_posterior_samples, ymax,
    title="Posterior predictive",
    label="Samples",
    ylabel="Passengers"
)
@df df scatter!(
    linear_posterior_plt, :Month, :Passengers,
    label="Data", color=2, legend=:topleft
)

# Plot trend
gq = generated_quantities(linear(t, missing), linear_posterior_chain)
trend = mapreduce(x -> x.trend, hcat, gq)
linear_posterior_trend_plt = plot(
    df.Month, trend*ymax,
    color=1,
    label=nothing,
    alpha=0.2,
    title="Posterior trend",
    xlabel="Month",
    ylabel="Passengers"
)
plot!(linear_posterior_trend_plt, [], [], color=1, label="Trend")
scatter!(
    linear_posterior_trend_plt,
    df.Month, df.Passengers,
    label="Data",
    color=2,
    legend=:topleft
)

linear_posterior_combined_plt = plot(
    vcat(linear_posterior_plt, linear_posterior_trend_plt)...,
    layout=(2, 1)
)
save_fig(linear_posterior_combined_plt, "linear_posterior_predictive.png")

function seasonality(x, order, period_length)
    period = x ./ period_length
    sines = Dict("sin$o" => sin.(2*π*o*period) for o in 1:order)
    coses = Dict("cos$o" => cos.(2*π*o*period) for o in 1:order)
    features = DataFrame(merge(sines, coses))
    Matrix(features)
end

order = 10
period = 365.25 # account for leap years
s = seasonality(dayofyear.(df.Month), order, period)

@model function linear_seasonality(t, s, y) 
    α ~ Normal(0, .5)
    βₜ ~ Normal(0, .5)
    βₛ ~ MvNormal(zeros(size(s, 2)), .1)
    σ ~ TruncatedNormal(0, .1, 0, Inf)
    
    seasonality = s * βₛ
    trend = α .+ βₜ .* t
    μ = trend .* (1 .+ seasonality)

    y ~ MvNormal(μ, σ)
    (; trend, seasonality)
end

function plot_linear_seasonality(df, gq, ymax) 
    trend = mapreduce(x -> x.trend, hcat, gq)
    seasonality = mapreduce(x -> x.seasonality, hcat, gq)
    
    trend_plt = plot(
        df.Month, trend*ymax,
        color=1,
        label=nothing,
        alpha=0.2,
        title="Trend",
        xlabel="Month",
        ylabel="Passengers"
    )

    idx2name = Dict(v => k for (k, v) in Dates.LOCALES["english"].month_value)
    seasonality_plt = plot(
        [idx2name[i] for i in month.(df.Month[1:12])],
        seasonality[1:12,:]*100,
        color=1,
        label=nothing,
        alpha=0.2,
        title="Seasonality",
        xlabel="Month",
        ylabel="Percentage change",
        xrotation=25
    )
    scatter!(
        trend_plt,
        df.Month, df.Passengers,
        label="Data",
        color=2,
        legend=:topleft
    )
    plot(trend_plt, seasonality_plt, layout=(2, 1))
end

linear_seasonality_prior_samples = sample(linear_seasonality(t, s, missing), Prior(), 100)
linear_seasonality_prior_plt = plot_samples(
    df.Month, linear_seasonality_prior_samples,
    ymax,
    label=nothing,
    title="Prior predictive",
    xlabel="Passengers"
)
@df df scatter!(
    linear_seasonality_prior_plt,
    :Month, :Passengers,
    label="Data",
    color=2,
    legend=:topleft
)

gq = generated_quantities(linear_seasonality(t, s, missing), linear_seasonality_prior_samples)
linear_sesonality_prior_decomposed_plt = plot_linear_seasonality(df, gq, ymax)
linear_seasonality_prior_combined_plt = plot(
    linear_seasonality_prior_plt, linear_sesonality_prior_decomposed_plt,
    size=(600, 800),
    layout=(2, 1)
)
save_fig(linear_seasonality_prior_combined_plt, "linear_seasonality_prior_predictive.png")


## Posterior predictive
linear_seasonal_posterior_chain = sample(linear_seasonality(t, s, y), NUTS(), 1000)
linear_seasonal_posterior_samples = predict(linear_seasonality(t, s, missing), linear_seasonal_posterior_chain)
linear_seasonal_posterior_plt = plot_samples(
    df.Month, linear_seasonal_posterior_samples, ymax,
    title="Posterior predictive",
    label="Samples",
    ylabel="Passengers"
)
@df df scatter!(
    linear_seasonal_posterior_plt, :Month, :Passengers,
    label="Data", color=2, legend=:topleft
)

gq = generated_quantities(linear_seasonality(t, s, missing), linear_seasonal_posterior_chain)
linear_posterior_decomposed_plt = plot_linear_seasonality(df, gq, ymax)

linear_seasonality_posterior_combined_plt = plot(
    vcat(linear_seasonal_posterior_plt, linear_posterior_decomposed_plt)...,
    layout=(2, 1),
    size=(600, 800)
)

save_fig(linear_seasonality_posterior_combined_plt, "linear_seasonality_posterior_predictive.png")


extended_months = vcat(df.Month, df.Month .- Year(minimum(df.Month)) .+ Year(maximum(df.Month)) .+ Month(maximum(df.Month)))
tmax = 12*year(maximum(df.Month)) + month(maximum(df.Month))
t̃ = prepare_t(extended_months, tmax)
s̃ = seasonality(dayofyear.(extended_months), order, period)

linear_seasonal_posterior_extended_samples = predict(
    linear_seasonality(t̃, s̃, missing),
    linear_seasonal_posterior_chain
)
linear_seasonal_posterior_extended_plt = plot_samples(
    extended_months, linear_seasonal_posterior_extended_samples, ymax,
    title="Posterior predictive forecast",
    label="Samples",
    ylabel="Passengers",
    xlabel="Month"
)
@df df scatter!(
    linear_seasonal_posterior_extended_plt, :Month, :Passengers,
    label="Data", color=2, legend=:topleft
)

save_fig(linear_seasonal_posterior_extended_plt, "linear_seasonality_posterior_forecast.png")