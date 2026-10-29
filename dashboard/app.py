"""MedBalance-Q v2 — Streamlit Dashboard

Calls the FastAPI backend at localhost:8000 for all data.
Run: streamlit run dashboard/app.py
"""

import pandas as pd
import requests
import streamlit as st

API = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_get(path: str) -> dict | None:
    try:
        r = requests.get(f"{API}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, body: dict) -> dict | None:
    try:
        r = requests.post(f"{API}{path}", json=body, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def status_color(stock: int, need: int) -> str:
    if stock == 0:
        return "red"
    if need > 0:
        return "yellow"
    return "green"


def status_emoji(stock: int, need: int) -> str:
    if stock == 0:
        return "🔴"
    if need > 0:
        return "🟡"
    return "🟢"


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MedBalance-Q v2",
    page_icon="💊",
    layout="wide",
)

st.title("MedBalance-Q v2")
st.caption("Expiry-Aware Essential Medicine Redistribution for District Health Networks")

# Check backend health
health = api_get("/health")
if health:
    st.sidebar.success(f"Backend: {health['status']} ({health['version']})")
else:
    st.sidebar.error("Backend offline — start with: uvicorn backend.main:app --port 8000")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "District Overview", "Stock Ingestion", "Generate Plan",
    "Quantum Panel", "Evaluation", "Evidence",
])


# =========================================================================
# Tab 1 — District Overview
# =========================================================================

with tab1:
    st.header("District Overview")

    inv = api_get("/inventory")
    if inv:
        facilities = inv["facilities"]
        stock = inv["facility_stock"]
        needs = inv["needs"]
        drugs = inv["drugs"]

        # --- Facility map ---
        fac_details = {}
        for fid in facilities:
            fac_inv = api_get(f"/inventory/{fid}")
            if fac_inv:
                fac_details[fid] = fac_inv

        FACILITY_COORDS = {
            "DS-01": (12.97, 77.59), "GH-01": (12.93, 77.58),
            "GH-02": (13.03, 77.65), "CHC-01": (12.85, 77.50),
            "CHC-02": (13.10, 77.72), "PHC-01": (12.78, 77.45),
            "PHC-02": (13.15, 77.80),
        }

        map_rows = []
        for fid in facilities:
            lat, lon = FACILITY_COORDS.get(fid, (0, 0))
            fac_stock_total = sum(
                v for k, v in stock.items() if k.startswith(f"{fid}|")
            )
            fac_need_total = sum(
                v for k, v in needs.items() if k.startswith(f"{fid}|")
            )
            map_rows.append({
                "lat": lat, "lon": lon,
            })

        if map_rows:
            st.map(pd.DataFrame(map_rows), use_container_width=True)

        # --- Status table ---
        st.subheader("Facility Status")
        rows = []
        for fid in facilities:
            fac_stock_total = sum(
                v for k, v in stock.items() if k.startswith(f"{fid}|")
            )
            fac_need_total = sum(
                v for k, v in needs.items() if k.startswith(f"{fid}|")
            )
            rows.append({
                "Status": status_emoji(fac_stock_total, fac_need_total),
                "Facility": fid,
                "Total Stock": fac_stock_total,
                "Total Need": fac_need_total,
                "Surplus Chunks": len(fac_details.get(fid, {}).get("surplus_chunks", [])),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # --- Drug-level breakdown ---
        st.subheader("Stock by Facility × Drug")
        drug_rows = []
        for fid in facilities:
            for dc in drugs:
                key = f"{fid}|{dc}"
                s = stock.get(key, 0)
                n = needs.get(key, 0)
                drug_rows.append({
                    "Facility": fid,
                    "Drug": dc,
                    "Stock": s,
                    "Need": n,
                    "Status": status_emoji(s, n),
                })
        df_drugs = pd.DataFrame(drug_rows)
        st.dataframe(df_drugs, use_container_width=True, hide_index=True, height=400)


# =========================================================================
# Tab 2 — Stock Ingestion
# =========================================================================

with tab2:
    st.header("Stock Ingestion")
    st.write("Paste a raw stock report. The GenAI layer extracts structured lot records.")

    col1, col2 = st.columns([2, 1])

    with col1:
        raw_report = st.text_area(
            "Raw Stock Report",
            height=200,
            placeholder=(
                "e.g.: Insulin 40 IU/ml - 3 vials, fridge. "
                "Metformin 500mg - 200 tabs. Amoxicillin 500mg - 15 caps."
            ),
        )

    with col2:
        facility_id = st.selectbox(
            "Facility",
            options=inv["facilities"] if inv else [],
        )
        parse_btn = st.button("Parse Report", type="primary")

    if parse_btn and raw_report:
        with st.spinner("Parsing with GenAI..."):
            result = api_post("/ingest", {
                "facility_id": facility_id,
                "raw_report": raw_report,
            })
        if result:
            accepted = result["accepted_lots"]
            flagged = result["flagged_lots"]

            if accepted:
                st.success(f"{len(accepted)} lots accepted")
                st.dataframe(pd.DataFrame(accepted), use_container_width=True, hide_index=True)
            else:
                st.info("No lots accepted (check API key or report format)")

            if flagged:
                st.warning(f"{len(flagged)} lots flagged for review")
                st.dataframe(
                    pd.DataFrame(flagged).style.map(
                        lambda v: "background-color: #fff3cd" if v else "",
                        subset=["flag_reason"],
                    ),
                    use_container_width=True, hide_index=True,
                )

            if result["confidence_scores"]:
                st.subheader("Confidence Scores")
                for cs in result["confidence_scores"]:
                    col_a, col_b = st.columns([3, 1])
                    col_a.write(cs["lot_id"])
                    col_b.progress(cs["confidence"], text=f"{cs['confidence']:.0%}")
    elif parse_btn:
        st.warning("Please paste a stock report first.")


# =========================================================================
# Tab 3 — Generate Plan
# =========================================================================

with tab3:
    st.header("Generate Redistribution Plan")

    col1, col2, col3 = st.columns(3)
    with col1:
        use_quantum = st.toggle("Use Quantum Solver", value=False)
    with col2:
        cover_days = st.slider("Cover Days", 7, 28, 14)
    with col3:
        solver = "opt" if use_quantum else "greedy"
        st.metric("Solver", solver.upper())

    gen_btn = st.button("Generate Plan", type="primary")

    if gen_btn:
        with st.spinner("Running pipeline..."):
            plan = api_post("/plan/generate", {
                "cover_days": cover_days,
                "use_quantum": use_quantum,
                "solver": solver,
            })
        if plan:
            st.session_state["current_plan"] = plan

    plan = st.session_state.get("current_plan")
    if plan:
        plan_id = plan["plan_id"]
        val = plan["validation"]

        # --- Summary metrics ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Plan ID", plan_id)
        m2.metric("Solver", plan["solver_used"])
        m3.metric("Transfers", len(plan["transfers"]))
        m4.metric("Wall Clock", f"{plan['wall_clock_s']:.2f}s")

        # --- Validation banner ---
        if val["is_valid"]:
            st.success("Validation passed — zero constraint violations")
        else:
            st.error(f"Validation failed: {len(val['violations'])} violations")
            for v in val["violations"]:
                st.write(f"- {v}")
        if val.get("warnings"):
            st.warning(f"{len(val['warnings'])} warnings")
        if val.get("fefo_deviations"):
            st.info(f"FEFO deviations: {len(val['fefo_deviations'])}")

        # --- Transfer orders ---
        if plan.get("orders"):
            st.subheader("Transfer Orders")
            for order in plan["orders"]:
                with st.expander(
                    f"**{order['order_id']}**: {order['quantity']} units of "
                    f"{order.get('drug_name', order['drug_code'])} — "
                    f"{order['source_id']} → {order['destination_id']} "
                    f"(expires in {order['days_to_expiry']}d)"
                ):
                    c1, c2 = st.columns(2)
                    c1.write(f"**Lot:** {order['lot_id']}")
                    c1.write(f"**Transport cost:** ₹{order.get('transport_cost', 0):.0f}")
                    c2.write(f"**Approved:** {'Yes' if order.get('approved') else 'No'}")

                    st.markdown("**Justification:**")
                    st.info(order["justification"])

                    if st.button("Show Alternatives", key=f"alt_{order['order_id']}"):
                        alts = api_get(f"/plan/{plan_id}/alternatives")
                        if alts and alts["entries"]:
                            st.write("**Regret Ledger Alternatives:**")
                            for e in alts["entries"][1:]:
                                st.write(
                                    f"- Rank {e['rank']}: "
                                    f"H={e['objective_value']:.3f} "
                                    f"(delta={e['delta_from_best']:+.3f}), "
                                    f"differs in {len(e['diff_from_chosen'])} transfers"
                                    f"{' [INFEASIBLE]' if not e['is_feasible'] else ''}"
                                )
                        else:
                            st.write("No alternatives recorded.")

            # --- Approve section ---
            st.subheader("Approve Orders")
            order_ids = [o["order_id"] for o in plan["orders"]]
            selected_orders = st.multiselect("Select orders to approve", order_ids)
            if st.button("Approve Selected") and selected_orders:
                result = api_post(f"/plan/{plan_id}/approve", {
                    "approved_order_ids": selected_orders,
                })
                if result:
                    st.success(f"Approved: {', '.join(result['approved'])}")
        else:
            st.info("No transfers needed — all facilities adequately stocked.")

        # --- Regret ledger summary ---
        st.subheader("Regret Ledger")
        st.code(plan["regret_ledger_summary"])


# =========================================================================
# Tab 4 — Quantum Panel
# =========================================================================

with tab4:
    st.header("Quantum Panel")

    st.write("Compare QCloud QAOA Max-Cut clustering against classical k-means.")

    qcol1, qcol2 = st.columns(2)

    with qcol1:
        if st.button("Run K-Means Only"):
            with st.spinner("Clustering..."):
                result = api_post("/clustering", {"method": "kmeans"})
            if result:
                st.session_state["clustering"] = result

    with qcol2:
        if st.button("Run Full Comparison (QAOA + K-Means)"):
            with st.spinner("Running QAOA + K-Means comparison..."):
                result = api_post("/clustering", {"method": "compare"})
            if result:
                st.session_state["clustering"] = result

    clust = st.session_state.get("clustering")
    if clust:
        st.subheader("Results")

        if clust.get("method") == "compare":
            comp = clust.get("comparison", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Winner", comp.get("winner", "N/A").upper())
            c2.metric("K-Means Cost", f"₹{comp.get('kmeans_cost', 0):,.0f}")
            qc_cost = comp.get("qcloud_cost")
            c3.metric("QAOA Cost", f"₹{qc_cost:,.0f}" if qc_cost is not None else "N/A")

            if comp.get("winner") == "kmeans":
                st.info(
                    "K-Means produced lower intra-cluster transport cost. "
                    "This is reported honestly — the classical method won on this instance."
                )
            elif clust.get("qaoa_available"):
                st.success("QAOA produced better clustering than k-means.")
            else:
                st.warning("QAOA was unavailable (auth required). Showing k-means only.")

            st.subheader("Chosen Clusters")
            for i, cluster in enumerate(clust.get("chosen_clusters", [])):
                st.write(f"**Cluster {i + 1}:** {', '.join(cluster) if cluster else '(empty)'}")

            st.subheader("K-Means Clusters")
            for i, cluster in enumerate(clust.get("kmeans_clusters", [])):
                st.write(f"**Cluster {i + 1}:** {', '.join(cluster) if cluster else '(empty)'}")

        elif clust.get("clusters"):
            st.write("**Clusters:**")
            for i, cluster in enumerate(clust["clusters"]):
                st.write(f"**Cluster {i + 1}:** {', '.join(cluster) if cluster else '(empty)'}")

            if clust.get("cut_value"):
                st.metric("Cut Value", f"{clust['cut_value']:.1f}")
            if clust.get("backend"):
                st.metric("Backend", clust["backend"])


# =========================================================================
# Tab 5 — Evaluation
# =========================================================================

with tab5:
    st.header("Evaluation")
    st.caption(
        "These results are from simulated scenarios on synthetic data. "
        "Daily usage varies ±20% to model real-world uncertainty."
    )

    ecol1, ecol2, ecol3 = st.columns(3)
    with ecol1:
        scenario = st.selectbox("Scenario", [
            "normal_week", "shortage_heavy", "expiry_heavy",
            "outbreak_spike", "cold_chain_down",
        ])
    with ecol2:
        n_seeds = st.number_input("Seeds", min_value=1, max_value=50, value=5)
    with ecol3:
        n_weeks = st.number_input("Weeks", min_value=1, max_value=8, value=4)

    if st.button("Run Replay", type="primary"):
        with st.spinner(f"Running {scenario} × {n_seeds} seeds × {n_weeks} weeks..."):
            result = api_post("/replay", {
                "scenario": scenario,
                "n_seeds": int(n_seeds),
                "n_weeks": int(n_weeks),
            })
        if result:
            st.session_state["replay"] = result

    replay = st.session_state.get("replay")
    if replay and "results" in replay:
        results = replay["results"]

        def _avg(runs, key):
            vals = [r[key] for r in runs]
            return sum(vals) / len(vals) if vals else 0

        summary_rows = []
        for mode in ("none", "greedy", "qubo"):
            runs = results.get(mode, [])
            if not runs:
                continue
            summary_rows.append({
                "Mode": mode.upper(),
                "Avg Stockout Days": f"{_avg(runs, 'stockout_days'):.1f}",
                "Avg Expired Units": f"{_avg(runs, 'expired_units'):.0f}",
                "Avg Expired Value (₹)": f"{_avg(runs, 'expired_value'):.0f}",
                "Avg Transport Cost (₹)": f"{_avg(runs, 'transport_cost'):.0f}",
                "Avg Transfers": f"{_avg(runs, 'transfers'):.0f}",
                "Avg Violations": f"{_avg(runs, 'violations'):.1f}",
            })

        st.subheader("Comparison Table")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # --- Bar charts ---
        st.subheader("Visualisation")

        chart_data = {}
        for mode in ("none", "greedy", "qubo"):
            runs = results.get(mode, [])
            if runs:
                chart_data[mode.upper()] = {
                    "Stockout Days": _avg(runs, "stockout_days"),
                    "Expired Units": _avg(runs, "expired_units"),
                }

        if chart_data:
            c1, c2 = st.columns(2)

            with c1:
                st.write("**Average Stockout Days**")
                df_so = pd.DataFrame({
                    "Mode": list(chart_data.keys()),
                    "Stockout Days": [v["Stockout Days"] for v in chart_data.values()],
                })
                st.bar_chart(df_so, x="Mode", y="Stockout Days")

            with c2:
                st.write("**Average Expired Units**")
                df_exp = pd.DataFrame({
                    "Mode": list(chart_data.keys()),
                    "Expired Units": [v["Expired Units"] for v in chart_data.values()],
                })
                st.bar_chart(df_exp, x="Mode", y="Expired Units")

        # Net value chart
        net_data = {}
        for mode in ("none", "greedy", "qubo"):
            runs = results.get(mode, [])
            if runs:
                net_data[mode.upper()] = (
                    _avg(runs, "expired_value") - _avg(runs, "transport_cost")
                )

        if net_data:
            st.write("**Net Value (Expiry Savings - Transport Cost)**")
            df_net = pd.DataFrame({
                "Mode": list(net_data.keys()),
                "Net Value (₹)": list(net_data.values()),
            })
            st.bar_chart(df_net, x="Mode", y="Net Value (₹)")


# =========================================================================
# Tab 6 — Evidence
# =========================================================================

with tab6:
    st.header("Evidence Files")
    st.write("All evidence produced by the pipeline is logged for auditability.")

    if st.button("Refresh Evidence List"):
        st.session_state.pop("evidence", None)

    evidence = st.session_state.get("evidence")
    if evidence is None:
        evidence = api_get("/evidence")
        if evidence:
            st.session_state["evidence"] = evidence

    if evidence and evidence.get("files"):
        files = evidence["files"]
        st.write(f"**{len(files)} evidence files found**")

        categories = {}
        for f in files:
            parts = f["path"].split("/")
            cat = parts[0] if len(parts) > 1 else "root"
            categories.setdefault(cat, []).append(f)

        for cat, cat_files in sorted(categories.items()):
            with st.expander(f"**{cat}/** ({len(cat_files)} files)"):
                for f in cat_files:
                    st.write(f"- `{f['path']}` ({f['size_bytes']:,} bytes)")
    else:
        st.info("No evidence files yet. Generate a plan or run a replay to create evidence.")
