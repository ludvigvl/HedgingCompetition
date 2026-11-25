#read the parquet file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import math
from datetime import date
from scipy.interpolate import PchipInterpolator


def build_fixing_schedule1(
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

    n_pairs = call_pairs.size
    price_call = D * call_pairs.mean()
    price_put = D * put_pairs.mean()
    std_call = D * call_pairs.std(ddof=1) / np.sqrt(n_pairs)
    std_put = D * put_pairs.std(ddof=1) / np.sqrt(n_pairs)

    if verbose:
        A_pairs = (A_plus + A_minus) / 2
        avg_se = A_pairs.std(ddof=1) / np.sqrt(n_pairs)
        print(f"Monte Carlo mean (call): {price_call:.6f}, SE: {std_call:.6f}")
        print(f"Monte Carlo mean (put): {price_put:.6f}, SE: {std_put:.6f}")
        print(f"Monte Carlo mean of A: {A_pairs.mean():.6f}, SE: {avg_se:.6f}")
    return price_call, price_put, std_call, std_put

def price_asian1(
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

    call_price, _, call_se, _ = price_asian1(K_call, **pricing_inputs)

    def premium_difference(K):
        _, put_price, _, _ = price_asian1(K, **pricing_inputs)
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
            _, put_price, _, _ = price_asian1(K, **put_inputs)
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

    _, put_price, _, put_se = price_asian1(K_put, **put_inputs)
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

MONTH_CODES = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
               'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}

def futures_expiry(contract):
    """ICE Brent futures expire on the last business day of the second month preceding delivery."""
    code = contract.split()[0]                      # 'COJ21 Comdty' -> 'COJ21'
    delivery = pd.Timestamp(2000 + int(code[3:5]), MONTH_CODES[code[2]], 1)
    return delivery - pd.DateOffset(months=1) - pd.offsets.BDay(1)

def build_fixing_schedule(year, month, front_contract):
    """Business days of the averaging month and number of fixings on the front / next contract."""
    start = pd.Timestamp(year, month, 1)
    fixing_dates = pd.bdate_range(start, start + pd.offsets.MonthEnd(0))

    # The front contract fixes up to and including the second-to-last fixing day before its expiry
    before_expiry = fixing_dates[fixing_dates < futures_expiry(front_contract)]
    last_front_fixing = before_expiry[-2]

    n1 = int((fixing_dates <= last_front_fixing).sum())
    return fixing_dates, n1, len(fixing_dates) - n1

def fixing_thetas(fixing_dates, n1, day):
    """Year fractions from `day` to each fixing, split by leg."""
    thetas = ((fixing_dates - pd.Timestamp(day)).days / 365.0).to_numpy()
    return list(thetas[:n1]), list(thetas[n1:])

def get_smile(day_data, contract, max_rel_spread=0.5, min_mid=0.05):
    """Implied vols from liquid out-of-the-money quotes, interpolated with PCHIP."""
    quotes = day_data[day_data['underlying_ticker'] == contract]
    quotes = quotes[quotes[['option_bid', 'option_ask']].notna().all(axis=1) | quotes['settlement_price'].notna()]
    if quotes.empty:
        return None
    F = quotes['underlying_settlement'].iloc[0]

    strikes, vols = [], []
    for _, ligne in quotes.iterrows():
        bid, ask, K = ligne['option_bid'], ligne['option_ask'], ligne['strike']

        if pd.notna(bid) and pd.notna(ask):
            prix = (bid + ask) / 2
            # Liquidity filter
            if bid <= 0 or ask <= bid or prix < min_mid or (ask - bid) / prix > max_rel_spread:
                continue
        else:
            # From June 2020 the file only has settlement prices (settlement = mid when both exist)
            prix = ligne['settlement_price']
            if prix < min_mid:
                continue

        # Out-of-the-money only: calls above F, puts below F
        if ligne['option_type'] == 'Call' and K >= F:
            prix_call = prix
        elif ligne['option_type'] == 'Put' and K < F:
            prix_call = prix + F - K        # put-call parity with r = 0
        else:
            continue

        T = ligne['time_to_expiry_days'] / 365
        vol = vol_implicite(prix_call, F, K, T)
        if 0.01 < vol < 4.9:                # drop vols stuck on the bisection bounds
            strikes.append(K)
            vols.append(vol)

    if len(strikes) < 3:
        return None
    order = np.argsort(strikes)
    strikes, vols = np.array(strikes)[order], np.array(vols)[order]
    return {'F': F, 'strikes': strikes, 'vols': vols,
            'pchip': PchipInterpolator(strikes, vols, extrapolate=False)}

def read_vol(smile, K):
    """Smile volatility at strike K, flat outside the quoted range."""
    K = min(max(K, smile['strikes'][0]), smile['strikes'][-1])
    return float(smile['pchip'](K))

def basket_variance(sigma1, sigma2, theta_1, theta_2, f1, f2, w1, w2, n1, n2, rho):
    """Same formulas as variance_totale, with rho passed explicitly."""
    var_A1 = sum((2 * (n1 - k) + 1) * (math.exp(sigma1**2 * theta_1[k-1]) - 1) for k in range(1, n1 + 1))
    var_A1 *= (f1**2 / n1**2)

    var_A2 = sum((2 * (n2 - j) + 1) * (math.exp(sigma2**2 * theta_2[j-1]) - 1) for j in range(1, n2 + 1))
    var_A2 *= (f2**2 / n2**2)

    cov = sum(math.exp(rho * sigma1 * sigma2 * theta_1[k-1]) - 1 for k in range(1, n1 + 1))
    cov *= (f1 * f2 / n1)

    return (w1**2 * var_A1) + (w2**2 * var_A2) + (2 * w1 * w2 * cov)

def price_asian(K, option_type, f1, f2, smile1, smile2, rho, theta_1, theta_2, D=1.0):
    """Moment-matching price of the Asian basket call/put, per barrel."""
    n1, n2 = len(theta_1), len(theta_2)
    w1, w2 = n1 / (n1 + n2), n2 / (n1 + n2)
    F_avg = w1 * f1 + w2 * f2

    sigma1 = read_vol(smile1, f1 * K / F_avg)
    sigma2 = read_vol(smile2, f2 * K / F_avg)

    var = basket_variance(sigma1, sigma2, theta_1, theta_2, f1, f2, w1, w2, n1, n2, rho)
    v = math.sqrt(math.log(1 + var / F_avg**2))
    return D * black_76(F_avg, K, v, 0, 1.0, option_type)

def solve_put_strike(call_price, f1, f2, smile1, smile2, rho, theta_1, theta_2, D=1.0, K_call=78.0):
    """Bisection on K_put such that Put(K_put) = Call(K_call)."""
    bas, haut = 1.0, K_call
    for _ in range(100):
        K_put = (bas + haut) / 2
        prix_put = price_asian(K_put, 'put', f1, f2, smile1, smile2, rho, theta_1, theta_2, D)
        if prix_put > call_price:
            haut = K_put
        else:
            bas = K_put
        if haut - bas < 1e-8:
            break
    return K_put

def discount_factor(day_data, day, payment_date):
    """SOFR discount factor (the dataset gives one overnight rate per day, in %)."""
    r = day_data['sofr_rate'].iloc[0] / 100
    return math.exp(-r * (pd.Timestamp(payment_date) - pd.Timestamp(day)).days / 365)

def estimate_rho_rolling(log_returns, tickers, day, window=60, min_obs=20, default=0.99):
    """Correlation of daily log-returns over the last `window` days up to `day`."""
    r = log_returns.loc[:day].dropna().tail(window)
    if len(r) < min_obs:
        return default
    return r[tickers[0]].corr(r[tickers[1]])


def collar_value(f1, f2, smile1, smile2, rho, theta_1, theta_2, D, K_put, K_call=78.0, notional=1_000_000):
    """Market maker's position in $: long Put(K_put), short Call(K_call)."""
    put = price_asian(K_put, 'put', f1, f2, smile1, smile2, rho, theta_1, theta_2, D)
    call = price_asian(K_call, 'call', f1, f2, smile1, smile2, rho, theta_1, theta_2, D)
    return notional * (put - call)

def collar_greeks(f1, f2, smile1, smile2, rho, theta_1, theta_2, D, K_put, bump=0.01):
    """Value, delta (bbl) and gamma (bbl per $) for a parallel move of both futures, by bump and reprice."""
    h = bump * f1
    args = (smile1, smile2, rho, theta_1, theta_2, D, K_put)
    V = collar_value(f1, f2, *args)
    V_up = collar_value(f1 + h, f2 + h, *args)
    V_down = collar_value(f1 - h, f2 - h, *args)
    delta = (V_up - V_down) / (2 * h)
    gamma = (V_up - 2 * V + V_down) / h**2
    return V, delta, gamma

def black_76_greeks(F, K, sigma, T, option_type='call'):
    """Closed-form Black-76 delta, gamma and vega per barrel (r = 0)."""
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (F * sigma * np.sqrt(T))
    vega = F * norm.pdf(d1) * np.sqrt(T)
    return delta, gamma, vega


def option_quote(day_data, contract, option_type, strike):
    """Price (mid, or settlement when there is no bid/ask), half bid-ask spread and time to expiry of a listed option."""
    q = day_data[(day_data['underlying_ticker'] == contract)
                 & (day_data['option_type'] == option_type)
                 & (day_data['strike'] == strike)]
    if q.empty:
        return None
    ligne = q.iloc[0]
    T = ligne['time_to_expiry_days'] / 365
    if pd.notna(ligne['option_bid']) and pd.notna(ligne['option_ask']):
        return (ligne['option_bid'] + ligne['option_ask']) / 2, (ligne['option_ask'] - ligne['option_bid']) / 2, T
    return ligne['settlement_price'], np.nan, T


def backtest_delta_gamma(greeks, df, smiles_by_day, futures_prices, contract, hedge_option,
                         use_gamma=True, half_spread_opt=0.12, half_spread_fut=0.03, band=0.0,
                         K_put=None, K_call=78.0):
    """Daily delta (and gamma) hedge of the collar with one futures contract and one listed option.
    hedge_option: ('Put', 45.0) for a fixed option, or 'auto' for the listed call closest to K_call when
    the collar is short gamma and the listed put closest to K_put when it is long gamma (round strikes,
    kept as long as the sign of the gamma does not change).
    half_spread_* are used on days without bid/ask quotes. Options are only traded when the position
    is more than `band` (in % of the target) away from the gamma-neutral quantity, or when the hedging
    option changes; futures are rebalanced daily."""
    days = greeks.index
    data_by_day = dict(tuple(df[df['date'].isin(days)].groupby('date')))

    held, n_opt, q_fut, cash = None, 0.0, 0.0, 0.0
    rows = []
    for i, day in enumerate(days):
        day_data = data_by_day[day]
        quotes = day_data[day_data['underlying_ticker'] == contract]
        smile = smiles_by_day[day][contract]
        F = futures_prices.at[day, contract]
        T = quotes['time_to_expiry_days'].iloc[0] / 365
        half_fut = (quotes['underlying_ask'].iloc[0] - quotes['underlying_bid'].iloc[0]) / 2
        if np.isnan(half_fut):
            half_fut = half_spread_fut

        def quote(option):
            """Price and half spread of an option; model price if the strike is not listed today."""
            q = option_quote(day_data, contract, *option)
            if q is None:
                return float(black_76(F, option[1], read_vol(smile, option[1]), 0, T, option[0].lower())), half_spread_opt
            prix, half, _ = q
            return prix, (half_spread_opt if np.isnan(half) else half)

        # 1. P&L of the positions held since yesterday (futures and options are settled daily in cash)
        pnl_collar = pnl_opt = pnl_fut = interest = 0.0
        if i > 0:
            pnl_collar = greeks.at[day, 'value'] - greeks.at[prev_day, 'value']
            if held is not None:
                pnl_opt = n_opt * (quote(held)[0] - prev_opt)
            pnl_fut = q_fut * (F - prev_F)
            interest = cash * prev_r * (day - prev_day).days / 365

        # 2. Hedging option of the day
        gamma_c = greeks.at[day, 'gamma']
        if hedge_option == 'auto':
            option_type, K_target = ('Call', K_call) if gamma_c < 0 else ('Put', K_put)
            # Round strikes only: they stay listed over the whole sample
            listed = quotes[(quotes['option_type'] == option_type) & (quotes['strike'] % 5 == 0)].dropna(subset=['settlement_price'])
            if held is not None and held[0] == option_type and held[1] in listed['strike'].values:
                target = held                      # same gamma sign: keep the option already held
            else:
                target = (option_type, float(listed['strike'].iloc[(listed['strike'] - K_target).abs().argmin()]))
        else:
            target = hedge_option
        delta_o, gamma_o, _ = black_76_greeks(F, target[1], read_vol(smile, target[1]), T, target[0].lower())
        n_target = -gamma_c / gamma_o if use_gamma else 0.0

        # 3. Option trades: a new option means closing the old position and opening the new one
        cost = 0.0
        if target != held:
            if held is not None:
                cost += abs(n_opt) * quote(held)[1]
            new_n = n_target
            cost += abs(new_n) * quote(target)[1]
            rebalance = True
        else:
            rebalance = abs(n_target - n_opt) > band * abs(n_target)
            new_n = n_target if rebalance else n_opt
            cost += abs(new_n - n_opt) * quote(target)[1]

        # 4. Futures cancel the remaining delta
        new_q = -(greeks.at[day, 'delta'] + new_n * delta_o)
        cost += abs(new_q - q_fut) * half_fut

        cash += pnl_opt + pnl_fut + interest - cost
        rows.append({'date': day, 'collar': greeks.at[day, 'value'], 'cash': cash,
                     'portfolio': greeks.at[day, 'value'] + cash,
                     'pnl': pnl_collar + pnl_opt + pnl_fut + interest - cost,
                     'pnl_collar': pnl_collar, 'pnl_options': pnl_opt, 'pnl_futures': pnl_fut,
                     'interest': interest, 'cost': cost,
                     'option': f"{target[0]} {target[1]:g}", 'n_options': new_n, 'q_futures': new_q,
                     'options_traded': rebalance and new_n != 0,
                     'residual_gamma': gamma_c + new_n * gamma_o})

        held, n_opt, q_fut = target, new_n, new_q
        prev_day, prev_opt, prev_F = day, quote(held)[0], F
        prev_r = day_data['sofr_rate'].iloc[0] / 100

    return pd.DataFrame(rows).set_index('date')


def asian_mc(
    K, option_type, futures, smiles, rho, theta, legs,
    fixed_sum, n_fixings, normal_draws, D=1.0,
):
    """Returnera pris och två sticky-vol-deltan, per fat."""
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    sign = 1.0 if option_type == "call" else -1.0
    theta = np.asarray(theta, dtype=float)
    legs = np.asarray(legs, dtype=int)
    futures = np.asarray(futures, dtype=float)

    if theta.size == 0:
        payoff = max(sign * (fixed_sum / n_fixings - K), 0.0)
        return D * payoff, np.zeros(2)

    active_legs = np.unique(legs)
    Favg = (fixed_sum + futures[legs].sum()) / n_fixings

    sigma_by_leg = np.zeros(2)
    for leg in active_legs:
        mapped_strike = K * futures[leg] / Favg
        sigma_by_leg[leg] = read_vol(smiles[leg], mapped_strike)

    sigma = sigma_by_leg[legs]
    mu = -0.5 * sigma**2 * theta
    correlation = np.where(legs[:, None] == legs[None, :], 1.0, rho)
    Sigma = (
        np.outer(sigma, sigma)
        * correlation
        * np.minimum.outer(theta, theta)
    )
    shock = np.linalg.cholesky(Sigma) @ normal_draws[:theta.size]

    price = 0.0
    deltas = np.zeros(2)

    for direction in (1, -1):
        multipliers = np.exp(mu[:, None] + direction * shock)
        simulated_fixings = futures[legs, None] * multipliers

        # A = w1*A1 + w2*A2 using the original fixing counts.
        # fixed_sum contains the observed fixings from both legs.
        A = (fixed_sum + simulated_fixings.sum(axis=0)) / n_fixings
        intrinsic = sign * (A - K)
        price += np.maximum(intrinsic, 0.0).mean()

        for leg in active_legs:
            dA_dF = multipliers[legs == leg].sum(axis=0) / n_fixings
            deltas[leg] += (sign * (intrinsic > 0) * dA_dF).mean()

    return D * price / 2, D * deltas / 2

def minimum_variance_position(
    futures_changes, collar_changes, delta_target,
    active, window=60, min_obs=20,
):
    """Returnera hedgepositioner i fat, antal observationer och metod."""
    active = np.asarray(active, dtype=bool)
    positions = np.zeros(2)

    if not active.any():
        return positions, 0, "settled"

    X = np.asarray(futures_changes, dtype=float).reshape(-1, 2)[:, active]
    y = np.asarray(collar_changes, dtype=float)

    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[valid][-window:], y[valid][-window:]

    if len(y) >= min_obs:
        X_centered = X - X.mean(axis=0)
        beta, _, rank, _ = np.linalg.lstsq(
            X_centered, y - y.mean(), rcond=1e-8
        )
        if rank == X_centered.shape[1]:
            positions[active] = -beta
            return positions, len(y), "minimum_variance"

    positions[active] = np.asarray(delta_target)[active]
    return positions, len(y), "delta_fallback"

def build_hedge_option_marks(df, option_specs):
    """Bygg dagliga marks och spreads för de valda noterade optionerna."""
    marks = {}

    for ticker, option_type, strike, fallback_half_spread in option_specs:
        rows = df.loc[
            df["underlying_ticker"].eq(ticker)
            & df["option_type"].str.lower().eq(option_type.lower())
            & df["strike"].eq(strike)
        ].copy()

        expiries = pd.to_datetime(rows["expiration"]).dropna().dt.normalize().unique()
        if len(expiries) != 1:
            raise ValueError(
                f"Förväntade en optionslöptid för {ticker} {option_type} {strike}, "
                f"hittade {len(expiries)}."
            )

        expiry = pd.Timestamp(expiries[0])
        bid = rows["option_bid"]
        ask = rows["option_ask"]
        valid_quote = np.isfinite(bid) & np.isfinite(ask) & bid.ge(0) & ask.ge(bid)
        settlement = rows["settlement_price"]
        valid_settlement = np.isfinite(settlement) & settlement.ge(0)

        rows["mark"] = settlement.where(valid_settlement).fillna(
            ((bid + ask) / 2).where(valid_quote)
        )
        rows["half_spread"] = ((ask - bid) / 2).where(
            valid_quote, fallback_half_spread
        )
        rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()

        name = f"{ticker} {option_type} {strike:g}"
        daily_marks = (
            rows.dropna(subset=["mark"])
            .drop_duplicates("date")
            .set_index("date")[["mark", "half_spread"]]
            .sort_index()
        )

        # Stäng senast handelsdagen före optionens förfall.
        close_by = expiry - pd.offsets.BDay(1)
        marks[name] = (daily_marks, close_by)

    return marks


def residual_minvar_position(
    instrument_changes, delta_residuals, baseline, eligible,
    window=60, min_obs=20, ridge=0.1,
):
    """Lägg en regulariserad minsta-varianskorrigering ovanpå delta."""
    baseline = np.asarray(baseline, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    target = baseline.copy()

    if not eligible.any():
        return target, 0, "delta_only"

    X = np.asarray(instrument_changes, dtype=float).reshape(-1, baseline.size)
    X = X[:, eligible]
    y = np.asarray(delta_residuals, dtype=float)

    # Begränsa först kalenderfönstret. Använd sedan kompletta observationer.
    X, y = X[-window:], y[-window:]
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]

    if len(y) < min_obs:
        return target, len(y), "delta_fallback"

    X = X - X.mean(axis=0)
    y = y - y.mean()
    covariance = X.T @ X / len(y)
    cross_covariance = X.T @ y / len(y)
    scale = np.trace(covariance) / covariance.shape[0]

    if scale <= 0:
        return target, len(y), "delta_fallback"

    penalty = ridge * scale * np.eye(covariance.shape[0])
    correction = np.linalg.solve(covariance + penalty, cross_covariance)
    target[eligible] -= correction

    return target, len(y), "residual_minvar"