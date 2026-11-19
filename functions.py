#read the parquet file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import math
from datetime import date
from scipy.interpolate import PchipInterpolator


def build_fixing_schedule(
    fixing_dates, valuation_date, front_expiry, front_contract, next_contract
):
    """Assign a futures contract and an ACT/365 time to each fixing."""
    fixing_dates = pd.DatetimeIndex(fixing_dates).sort_values()
    valuation_date = pd.Timestamp(valuation_date)
    front_expiry = pd.Timestamp(front_expiry)

    if fixing_dates.hasnans or fixing_dates.has_duplicates:
        raise ValueError("Fixing dates must be valid and unique.")

    pre_expiry_fixings = fixing_dates[fixing_dates < front_expiry]
    if len(pre_expiry_fixings) < 2:
        raise ValueError("Annex B requires at least two fixings before expiry.")

    # The front contract applies through the second-to-last pre-expiry fixing.
    last_front_fixing = pre_expiry_fixings[-2]
    uses_front = fixing_dates <= last_front_fixing

    return pd.DataFrame({
        "fixing_date": fixing_dates,
        "contract": pd.Series(uses_front).map({
            True: front_contract,
            False: next_contract,
        }),
        "theta": (fixing_dates - valuation_date).days / 365,
    })

def calibrate_day(day_quotes, futures_history, calibration_date):
    """Calibrate the Apr21 and May21 smiles and correlation for one date."""
    result = {"rho": estimate_rho(futures_history, calibration_date)}

    for ticker, leg in [("COJ21 Comdty", "apr21"), ("COK21 Comdty", "may21")]:
        quotes = day_quotes.loc[day_quotes["underlying_ticker"].eq(ticker)]

        settlements = quotes["underlying_settlement"].dropna()
        settlements = settlements[np.isfinite(settlements) & settlements.gt(0)]
        F = float(settlements.iloc[0]) if len(settlements) else np.nan

        result[f"F_{leg}"] = F
        result[f"smile_{leg}"] = None
        result[f"n_strikes_{leg}"] = 0

        if not np.isfinite(F):
            result[f"status_{leg}"] = "missing settlement"
            continue

        expiry_days = (pd.to_datetime(quotes["expiration"]) - calibration_date).dt.days
        calls = quotes.loc[
            quotes["option_type"].eq("Call")
            & quotes["option_bid"].gt(0)
            & quotes["option_ask"].ge(quotes["option_bid"])
            & quotes["strike"].gt(0)
            & expiry_days.gt(0)
        ].assign(T=expiry_days / 365)

        calls = calls.loc[
            np.isfinite(calls[["strike", "option_bid", "option_ask"]]).all(axis=1)
        ]

        points = []
        for quote in calls.itertuples():
            mid = (quote.option_bid + quote.option_ask) / 2
            try:
                vol = implied_volatility(F, quote.strike, quote.T, 0, mid, tol=1e-6)
            except ValueError:
                # No Black–76 implied volatility within the solver's bracket.
                continue
            points.append((quote.strike, vol))

        curve = (
            pd.DataFrame(points, columns=["strike", "vol"])
            .groupby("strike")["vol"]
            .median()
            .sort_index()
        )

        result[f"n_strikes_{leg}"] = len(curve)
        if len(curve) >= 2:
            result[f"smile_{leg}"] = PchipInterpolator(
                curve.index, curve.to_numpy(), extrapolate=True
            )
            result[f"status_{leg}"] = "ok"
        else:
            result[f"status_{leg}"] = "fewer than two valid call strikes"

    return result


def Phi(x):
    """Cumulative distribution function of the standard normal distribution."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def black_76(F, K, sigma, t, T, option_type='call'):
    """Black-76 formula for pricing European Call or Put options (interest rate r = 0)."""
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)

    if option_type == 'call':
        return F * norm.cdf(d1) - K * norm.cdf(d2)
    elif option_type == 'put':
        return K * norm.cdf(-d2) - F * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type")

def vol_implicite(prix, F, K, T):
    """Extracts the implied volatility of an option using the bisection method."""
    bas, haut = 0.0001, 5.0
    for _ in range(100):  # Safety iteration limit to prevent infinite loops
        milieu = (bas + haut) / 2
        if black_76(F, K, milieu, 0, T, option_type='call') < prix:
            bas = milieu
        else:
            haut = milieu
        if haut - bas < 1e-6:
            break
    return (bas + haut) / 2


def implied_volatility(f, K, T, t, market_price, option_type='call', tol=1e-5, max_iterations=100):
    objective_function = lambda sigma: black_76(f, K, sigma, t, T, option_type) - market_price
    return brentq(objective_function, 1e-6, 5, xtol=tol, maxiter=max_iterations)

def get_volatility(F1,F2,Favg,K, smile_apr21, smile_may21):
    K1 = K/Favg*F1
    K2 = K/Favg*F2

    sigma_apr = float(smile_apr21(K1))
    sigma_may = float(smile_may21(K2))
    return sigma_apr, sigma_may


# 2. Dynamic estimation of the correlation rho up to the valuation date
def estimate_rho(data, recalibration_date):
    """Computes the historical correlation of log-returns between April and May futures."""
    tickers = ["COJ21 Comdty", "COK21 Comdty"]
    
    prices = (
        data.loc[data["date"] <= recalibration_date]
        [['date', 'underlying_ticker', 'underlying_settlement']]
        .drop_duplicates(subset=["date", "underlying_ticker"])
        .pivot(index="date", columns="underlying_ticker", values="underlying_settlement")
        .sort_index()
        [tickers]
        .dropna()
    )
    
    returns = np.log(prices / prices.shift(1)).dropna()
    
    # Fallback to a default high correlation if historical window is too short (e.g., initial valuation day)
    if len(returns) < 2:
        return 0.99  
        
    return returns["COJ21 Comdty"].corr(returns["COK21 Comdty"])

# 5. Moment-matching engine (Variances and Cross-Covariance)
def variance_totale(sigma1, sigma2, theta_1, theta_2, f1, f2, w1, w2, n1, n2,rho):
    var_A1 = sum((2 * (n1 - k) + 1) * (math.exp(sigma1**2 * theta_1[k-1]) - 1) for k in range(1, n1 + 1))
    var_A1 *= (f1**2 / n1**2)

    var_A2 = sum((2 * (n2 - j) + 1) * (math.exp(sigma2**2 * theta_2[j-1]) - 1) for j in range(1, n2 + 1))
    var_A2 *= (f2**2 / n2**2)

    cov = sum(math.exp(rho * sigma1 * sigma2 * theta_1[k-1]) - 1 for k in range(1, n1 + 1))
    cov *= (f1 * f2 / n1)

    var_totale = (w1**2 * var_A1) + (w2**2 * var_A2) + (2 * w1 * w2 * cov)
    return var_totale



def simulate_correlated_futures(
    F1, F2,
    sigma1, sigma2,
    rho, theta, n1, D, K=78,
    n_simulations=10000,*,
    normal_draws=None,
    verbose=True,
):
    
    N = len(theta)
    n2 = N - n1
    w1,w2 = n1/N, (N-n1)/N
    sigma = np.where(
        np.arange(N) < n1,
        sigma1,
        sigma2
    )

    mu = -0.5 * sigma**2 * theta
    min_theta = np.minimum.outer(theta, theta)

    Sigma = np.zeros((N, N))

    for a in range(N):
        for b in range(N):
            #minimum of theta[a] and theta[b]
            min_theta = min(theta[a], theta[b])
            if a < n1 and b < n1:
            # Both fixings use Apr21
                Sigma[a, b] = sigma1**2 * min_theta
            elif a >= n1 and b >= n1:
            # Both fixings use May21
                Sigma[a, b] = sigma2**2 * min_theta
            else:
            # One fixing uses Apr21, the other May21
                Sigma[a, b] = rho * sigma1 * sigma2 * min_theta

    Chol = np.linalg.cholesky(Sigma)
    if normal_draws is None:
        Z = np.random.normal(size=(N, n_simulations))
    else:
        Z = np.asarray(normal_draws)
        if Z.shape != (N, n_simulations):
            raise ValueError(f"normal_draws must have shape {(N, n_simulations)}")
    
    F0 = np.where(np.arange(N) < n1, F1, F2)
    L = np.linalg.cholesky(Sigma)
    shock = L @ Z

    # Simulate each branch, then average over its fixing dates.
    fixings_plus = F0[:, None] * np.exp(mu[:, None] + shock)
    fixings_minus = F0[:, None] * np.exp(mu[:, None] - shock)



    # Apr21 arithmetic average on each branch
    A1_plus = fixings_plus[:n1, :].mean(axis=0)
    A1_minus = fixings_minus[:n1, :].mean(axis=0)

    # May21 arithmetic average on each branch
    A2_plus = fixings_plus[n1:, :].mean(axis=0)
    A2_minus = fixings_minus[n1:, :].mean(axis=0)

    # Combine the two futures legs within each branch
    A_plus = w1 * A1_plus + w2 * A2_plus
    A_minus = w1 * A1_minus + w2 * A2_minus

    call_plus = np.maximum(A_plus - K, 0.0)
    call_minus = np.maximum(A_minus - K, 0.0)
    put_plus = np.maximum(K - A_plus, 0.0)
    put_minus = np.maximum(K - A_minus, 0.0)

    call_pairs = (call_plus + call_minus) / 2
    put_pairs = (put_plus + put_minus) / 2

    
    print(f"Monte Carlo mean (call): {call_pairs.mean():.6f}, Monte Carlo std (call): {call_pairs.std(ddof=1)/np.sqrt(call_pairs.size):.6f}")
    print(f"Monte Carlo mean (put): {put_pairs.mean():.6f}, Monte Carlo std (put): {put_pairs.std(ddof=1)/np.sqrt(put_pairs.size):.6f}")

    n_pairs = call_pairs.size
    price_call = D * call_pairs.mean()
    price_put = D * put_pairs.mean()
    std_call = D * call_pairs.std(ddof=1) / np.sqrt(n_pairs)
    std_put = D * put_pairs.std(ddof=1) / np.sqrt(n_pairs)

    if verbose:
        A_pairs = (A_plus + A_minus) / 2
        avg_se = A_pairs.std(ddof=1) / np.sqrt(n_pairs)
        print(f"Monte Carlo mean of A: {A_pairs.mean():.6f}, SE: {avg_se:.6f}")
    return price_call, price_put, std_call, std_put

def price_asian(
    K, *,
    F1, F2, Favg,
    smile_apr21, smile_may21,
    rho, theta, n1,
    normal_draws,
    D=1.0,
):
    sigma1, sigma2 = get_volatility(
        F1, F2, Favg, K, smile_apr21, smile_may21
    )

    if not np.all(np.isfinite([sigma1, sigma2])) or min(sigma1, sigma2) <= 0:
        raise ValueError(f"Invalid smile volatility at basket strike {K:.6f}")

    return simulate_correlated_futures(
        F1=F1,
        F2=F2,
        sigma1=sigma1,
        sigma2=sigma2,
        rho=rho,
        theta=theta,
        n1=n1,
        D=D,
        K=K,
        n_simulations=normal_draws.shape[1],
        normal_draws=normal_draws,
        verbose=False,
    )

def find_strike_bracket(objective, lower, upper, n_grid=41):
    """Find the first sign change on an increasing strike grid."""
    strikes = np.linspace(lower, upper, n_grid)
    gaps = np.array([objective(K) for K in strikes])

    for i, gap in enumerate(gaps):
        if gap == 0.0:
            return (strikes[i], strikes[i]), gaps

        if i > 0 and np.signbit(gaps[i - 1]) != np.signbit(gap):
            return (strikes[i - 1], strikes[i]), gaps

    return None, gaps


def flat_lower_wing(smile):
    """Hold implied volatility constant below this leg's lowest strike."""
    lowest_strike = smile.x[0]

    def evaluate(strike):
        return smile(max(strike, lowest_strike))

    return evaluate


def solve_zero_cost_collar(
    *,
    F1, F2,
    smile_apr21, smile_may21,
    rho, theta, n1,
    K_call=78.0,
    n_simulations=100_000,
    seed=42,
    D=1.0,
):
    n_fixings = len(theta)
    Favg = (n1 * F1 + (n_fixings - n1) * F2) / n_fixings

    # Common basket-strike range implied by the two calibrated smiles.
    K_low = max(
        smile_apr21.x[0] * Favg / F1,
        smile_may21.x[0] * Favg / F2,
    ) + 1e-8
    K_high = min(
        smile_apr21.x[-1] * Favg / F1,
        smile_may21.x[-1] * Favg / F2,
    ) - 1e-8

    if K_low >= K_high:
        raise ValueError("The calibrated smiles have no common strike range.")

    if not K_low <= K_call <= K_high:
        raise ValueError(
            f"Call strike {K_call:g} is outside the common smile range "
            f"[{K_low:.4f}, {K_high:.4f}]."
        )

    normal_draws = np.random.default_rng(seed).standard_normal(
        (n_fixings, n_simulations)
    )

    pricing_inputs = dict(
        F1=F1,
        F2=F2,
        Favg=Favg,
        smile_apr21=smile_apr21,
        smile_may21=smile_may21,
        rho=rho,
        theta=theta,
        n1=n1,
        normal_draws=normal_draws,
        D=D,
    )

    call_price, _, call_se, _ = price_asian(K_call, **pricing_inputs)

    def premium_difference(K):
        _, put_price, _, _ = price_asian(K, **pricing_inputs)
        return put_price - call_price

    bracket, gaps = find_strike_bracket(premium_difference, K_low, K_high)
    objective = premium_difference
    put_inputs = pricing_inputs
    used_flat_lower_wing = False

    if bracket is None:
        if not np.all(gaps > 0):
            raise ValueError(
                "No put strike was bracketed on the calibrated smile grid. "
                f"Observed premium gaps: {gaps.min():+.6f} to "
                f"{gaps.max():+.6f} USD/bbl."
            )

        put_inputs = {
            **pricing_inputs,
            "smile_apr21": flat_lower_wing(smile_apr21),
            "smile_may21": flat_lower_wing(smile_may21),
        }

        def lower_wing_difference(K):
            _, put_price, _, _ = price_asian(K, **put_inputs)
            return put_price - call_price

        objective = lower_wing_difference
        bracket, _ = find_strike_bracket(objective, 0.0, K_low)

        if bracket is None:
            raise ValueError(
                "No put strike was bracketed below the calibrated range."
            )

        used_flat_lower_wing = True

    K_left, K_right = bracket
    if K_left == K_right:
        K_put = K_left
    else:
        K_put = brentq(objective, K_left, K_right, xtol=1e-6)

    _, put_price, _, put_se = price_asian(K_put, **put_inputs)
    return {
        "Call strike (USD/bbl)": K_call,
        "Zero-cost put strike (USD/bbl)": K_put,
        "Call premium (USD/bbl)": call_price,
        "Put premium (USD/bbl)": put_price,
        "Put minus call premium (USD/bbl)": put_price - call_price,
        "Call MC standard error (USD/bbl)": call_se,
        "Put MC standard error (USD/bbl)": put_se,
        "Used flat lower-wing extrapolation": used_flat_lower_wing,
    }