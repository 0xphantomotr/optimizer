import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, value
from datetime import datetime
import os
import warnings
import platform

warnings.filterwarnings("ignore", category=UserWarning)

# Battery Parameters added the P_Gridmax parameted and also the K (max number of charges and discharges)
BATTERY_CAPACITY = 9.0
BATTERY_EFFICIENCY = 0.90
MAX_CHARGE_RATE = 9.0
MAX_DISCHARGE_RATE = 9.0
INITIAL_SOC = 0.2
MIN_SOC = 0.2
MAX_SOC = 0.98
P_GRIDMAX = 5
K = 15
PERCENT_POWER = 1/BATTERY_CAPACITY

def get_all_days(conn):
    q = """
    SELECT DISTINCT date AS day_str, to_date(date, 'MM/DD/YYYY') AS sort_key
    FROM consumptionforecast
    WHERE date ~ '^[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}$'
    ORDER BY sort_key
    """
    df = pd.read_sql(q, conn)
    return df["day_str"].tolist()

def format_date_for_pv(day_str):
    try:
        dt = datetime.strptime(day_str, "%m/%d/%Y")
        return f"{dt.month}/{dt.day}/{dt.year}"  # always returns like 1/2/2025
    except Exception as e:
        print(f"❌ format_date_for_pv error for '{day_str}': {e}")
        return None
    
def pad_to_24_hours(df, column_defaults):
    full_hours = pd.DataFrame({"hour": list(range(1, 25))})
    df = full_hours.merge(df, on="hour", how="left")
    for col, default in column_defaults.items():
        df[col] = df[col].fillna(default)
    return df

def load_day_data(conn, day_str):
    print(f"🔍 Loading data for day: {day_str}")
    
    try:
        q1 = f"""
        SELECT hour, consumption_kw::float AS load, reference_tariff AS segment, month
        FROM consumptionforecast
        WHERE date = '{day_str}'
        ORDER BY hour
        """
        df_load = pd.read_sql(q1, conn)
        print(f"📊 Consumption rows: {len(df_load)}")
    except Exception as e:
        print(f"❌ Failed loading consumptionforecast: {e}")
        return None

    if df_load.empty:
        print("⚠️ df_load is empty")
        return None

    df_load["lookup"] = df_load["month"] + df_load["segment"]

    try:
        q_tariffs = "SELECT lookup, total_variable_cost_below_150 AS grid_price FROM monthly_tariffs"
        tariff_df = pd.read_sql(q_tariffs, conn)
        df_load = df_load.merge(tariff_df, on="lookup", how="left")
    except Exception as e:
        print(f"❌ Failed loading tariffs: {e}")
        return None

    pv_date = format_date_for_pv(day_str)
    print(f"🗓️ Converted PV date: {pv_date}")
    if not pv_date:
        print("⚠️ Invalid pv_date")
        return None

    try:
        q2 = f"""
        SELECT hour::int, power_corrected AS pv_raw
        FROM pvdata
        WHERE date = '{pv_date}'
        ORDER BY hour
        """
        df_pv = pd.read_sql(q2, conn)
        print(f"📈 PV rows: {len(df_pv)}")
    except Exception as e:
        print(f"❌ Failed loading pvdata: {e}")
        return None

    df_pv["pv"] = pd.to_numeric(df_pv["pv_raw"], errors="coerce")
    df_pv = df_pv.drop(columns=["pv_raw"]).dropna(subset=["pv"])
    if df_pv.empty:
        print("⚠️ df_pv is empty after cleaning")
        return None

    df = pd.merge(df_load, df_pv, on="hour", how="outer").sort_values("hour")
    df["GridPrice"] = df["grid_price"]

    # Fill missing hours
    df = pad_to_24_hours(df, {
        "load": 0,
        "pv": 0,
        "GridPrice": df["grid_price"].mean() if "grid_price" in df and df["grid_price"].notna().any() else 0.2,
        "lookup": "",
        "segment": "",
        "month": ""
    })

    print(f"✅ Day data ready: {len(df)} rows")
    return df

def solve_day(df, init_soc, max_cycles=2):
    T = len(df)
    model = LpProblem("EMS_Optimization", LpMinimize)

    # Decision variables Pg and Soc gone
    
    Pg_load = [LpVariable(f"Pg_load_{t}", lowBound=0) for t in range(T)]
    Pg_batt = [LpVariable(f"Pg_batt_{t}", lowBound=0) for t in range(T)]
    Ppv_load = [LpVariable(f"Ppv_load_{t}", lowBound=0) for t in range(T)]
    Ppv_batt = [LpVariable(f"Ppv_batt_{t}", lowBound=0) for t in range(T)]
    P_ch = [LpVariable(f"P_ch_{t}", lowBound=0) for t in range(T)]  # charge into battery
    P_dis = [LpVariable(f"P_dis_{t}", lowBound=0) for t in range(T)]  # discharge from battery
    Pg = [LpVariable(f"Pg_{t}", lowBound=0) for t in range(T)]
    SOC = [LpVariable(f"SoC_{t}", lowBound=MIN_SOC, upBound=MAX_SOC) for t in range(T)]

    # Binary control variables OK
    ych = [LpVariable(f"ych_{t}", cat=LpBinary) for t in range(T)]
    ydis = [LpVariable(f"ydis_{t}", cat=LpBinary) for t in range(T)]
    alpha = [LpVariable(f"alpha_{t}", cat=LpBinary) for t in range(T)]
    lambd = [LpVariable(f"lambda_{t}", cat=LpBinary) for t in range(1, T)]

    # Equation 21 
    model += lpSum(Pg[t] * df.loc[t, "GridPrice"] for t in range(T))

    for t in range(T):
        load = df.loc[t, "load"]
        pv = df.loc[t, "pv"]  # make sure pv is defined first
        
        model += Pg[t] == Pg_load[t] + Pg_batt[t]
        model += Ppv_load[t] + Pg[t] + P_dis[t] - P_ch[t] == load

        # Equation (6) ok
        model += Pg[t] == Pg_load[t] + Pg_batt[t]

        # Equation (7) ok
        model += Ppv_load[t] + Ppv_batt[t] <= pv

        # Equation (8) ok
        model += P_ch[t] == Pg_batt[t] + Ppv_batt[t]

        # Disallow simultaneous charge/discharge Euqation (3)
        model += ych[t] + ydis[t] <= 1

              
         # Charging constraints equation (4)
        model += P_ch[t] <= MAX_CHARGE_RATE * ych[t]
        # CHARGING LIMIT: in the SOC
    

        # Discharging constraints Euqation (4)
        model += P_dis[t] <= MAX_DISCHARGE_RATE * ydis[t]

        # Power from grid below maximum power (5)

        for t in range(T):
           model += Pg_load[t] + Pg_batt[t] <= P_GRIDMAX
    
        # Equation (9) Battery energy to load

        for t in range(T):
           model += BATTERY_CAPACITY * SOC[t] + P_ch[t] * BATTERY_EFFICIENCY - P_dis[t] * BATTERY_EFFICIENCY <= BATTERY_CAPACITY
    
        # Equation 13 initial SOC at time 1

        model += SOC[0] == init_soc
   


    for t in range(T - 1):
        model += SOC[t + 1] == SOC[t] + (P_ch[t] * BATTERY_EFFICIENCY * PERCENT_POWER - P_dis[t] * BATTERY_EFFICIENCY * PERCENT_POWER)

        # Constraints (15–18): Cycle change detection
        model += lambd[t - 1] >= alpha[t] - alpha[t - 1]
        model += lambd[t - 1] >= alpha[t - 1] - alpha[t]
        model += lambd[t - 1] <= alpha[t] + alpha[t - 1]
        model += lambd[t - 1] <= 2 - alpha[t] - alpha[t - 1]

    # Constraint (19): limit the total number of cycles
    model += lpSum(lambd) <= K

    model.solve()

    # Extract results
    df["Grid"] = [value(Pg[t]) for t in range(T)]
    df["Pg_load"] = [value(Pg_load[t]) for t in range(T)]
    df["Pg_batt"] = [value(Pg_batt[t]) for t in range(T)]
    df["Ppv_load"] = [value(Ppv_load[t]) for t in range(T)]
    df["Ppv_batt"] = [value(Ppv_batt[t]) for t in range(T)]
    df["P_ch"] = [value(P_ch[t]) for t in range(T)]
    df["P_dis"] = [value(P_dis[t]) for t in range(T)]
    df["SoC"] = [value(SOC[t]) for t in range(T)]
    df["hour"] = list(range(1, 25))
    df["Day"] = df.get("Day", [""] * T)

    # Derived values
    df["Pch_from_grid"] = df["Pg_batt"]
    df["Pch_from_pv"] = df["Ppv_batt"]
    df["GridCost_LoadOnly"] = df["Pg_load"] * df["GridPrice"]
    df["GridCost_BattOnly"] = df["Pg_batt"] * df["GridPrice"]
    df["BattChargeCost"] = df["Pch_from_grid"] * df["GridPrice"]
    df["Cost"] = df["Grid"] * df["GridPrice"]
    df["Total_PV_used"] = df["Ppv_load"] + df["Ppv_batt"]
    df["Total_Grid_used"] = df["Pg_load"] + df["Pg_batt"]
    df["Total_Load_Supplied"] = df["Ppv_load"] + df["P_dis"] + df["Pg_load"]
    df["is_charging"] = df["P_ch"] > 0
    df["is_discharging"] = df["P_dis"] > 0

    return df, value(SOC[-1])

def plot_graphs(df, outdir, day_label):
    os.makedirs(outdir, exist_ok=True)
    safe_label = day_label.replace("/", "_")

    fig, ax = plt.subplots(figsize=(10, 5))

    # Bars for charging/discharging
    ax.bar(df["hour"], df["P_ch"], color="green", alpha=0.6, label="Batt(Ch)")
    ax.bar(df["hour"], -df["P_dis"], color="orange", alpha=0.6, label="Batt(Disch)")

    # Power flow lines
    ax.plot(df["hour"], df["load"], "--", color="brown", label="Load")
    ax.plot(df["hour"], df["Pg_load"] + df["Pg_batt"], "c--", label="Grid Power")
    ax.plot(df["hour"], df["Ppv_load"] + df["Ppv_batt"], "k--", label="Utilized PV")

    # SoC on second y-axis
    ax2 = ax.twinx()
    ax2.plot(df["hour"], df["SoC"], "r-", linewidth=2, label="BatSOC")

    # Labels and legends
    ax.set_title(f"Dispatch Profile: {day_label}")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (kW)")
    ax2.set_ylabel("State of Charge (SoC)")

    # Combine legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{outdir}/dispatch_{safe_label}.png")
    plt.close()
    print(f"✅ Saved: {outdir}/dispatch_{safe_label}.png")

def plot_monthly_summary(df, outdir="graphs/monthly"):
    os.makedirs(outdir, exist_ok=True)
    df["Month"] = pd.to_datetime(df["Day"]).dt.strftime("%B")
    df["hour"] = df["hour"].astype(int)

    grouped = df.groupby(["Month", "hour"]).mean(numeric_only=True).reset_index()
    months = sorted(grouped["Month"].unique(), key=lambda m: datetime.strptime(m, "%B").month)

    for month in months:
        sub = grouped[grouped["Month"] == month]

        fig, ax = plt.subplots(figsize=(10, 5))

        # Bars for charging/discharging
        ax.bar(sub["hour"], sub["P_ch"], color="green", alpha=0.6, label="Batt(Ch)")
        ax.bar(sub["hour"], -sub["P_dis"], color="orange", alpha=0.6, label="Batt(Disch)")

        # Power flow lines
        ax.plot(sub["hour"], sub["load"], "--", color="brown", label="Load")
        ax.plot(sub["hour"], sub["Pg_load"] + sub["Pg_batt"], "c--", label="Grid Power")
        ax.plot(sub["hour"], sub["Ppv_load"] + sub["Ppv_batt"], "k--", label="Utilized PV")

        # SoC on second y-axis
        ax2 = ax.twinx()
        ax2.plot(sub["hour"], sub["SoC"], "r-", linewidth=2, label="BatSOC")

        # Labels and legends
        ax.set_title(f"{month}")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Power (kW)")
        ax2.set_ylabel("State of Charge (SoC)")

        # Combine legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right")

        plt.tight_layout()
        filename = f"{outdir}/dispatch_{month}.png"
        plt.savefig(filename)
        plt.close()
        print(f"✅ Saved: {filename}")

db_url = os.getenv("DATABASE_URL")

def main():
    conn = psycopg2.connect(db_url)

    all_days = get_all_days(conn)
    soc = INITIAL_SOC
    results = []
    skipped_days = []

    for day in all_days:
        df = load_day_data(conn, day)
        if df is None or df.empty:
            skipped_days.append(day)
            continue
        solved, soc = solve_day(df, soc)
        solved["Day"] = day
        results.append(solved)
        plot_graphs(solved, "graphs/daily", day)

    conn.close()

    print(f"\n✅ Optimization complete")
    print(f"Processed: {len(results)} days")
    print(f"Skipped:   {len(skipped_days)} days")

    if not results:
        print("⚠️ No valid days found — exiting.")
        return

    final = pd.concat(results)
    final["Cost"] = final["Grid"] * final["GridPrice"]
    final.to_csv("yearly_hourly_dispatch.csv", index=False)
    print("✅ Saved: yearly_hourly_dispatch.csv")

    # Daily bar chart
    daily = final.groupby("Day")["Cost"].sum().reset_index()
    daily["DayIndex"] = range(len(daily))
    plt.figure(figsize=(12, 4))
    plt.bar(daily["DayIndex"], daily["Cost"], color="orange")
    plt.title("Daily Operating Cost")
    plt.xlabel("Day Index")
    plt.ylabel("Cost (€)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("graphs/daily_cost.png")
    print("✅ Saved: graphs/daily_cost.png")

    plot_monthly_summary(final)

if __name__ == "__main__":
    main()