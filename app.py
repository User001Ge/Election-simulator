from __future__ import annotations

import random

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from election_engine import (
    RUNOFF_LABEL,
    load_model_data,
    run_monte_carlo,
    simulate_single_election,
)


st.set_page_config(
    page_title="საპატრიარქო არჩევნების სიმულაცია",
    page_icon="🗳️",
    layout="wide",
)

BAR_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
MAX_VOTERS = 39


@st.cache_resource
def get_model():
    return load_model_data()


MODEL = get_model()


@st.cache_data(show_spinner=False)
def get_monte_carlo_result(
    selected_candidates: tuple[str, str, str],
    iterations: int,
    voter_absence_probability: float,
    volatility_level: int,
    candidate_absence_probability: float,
    enable_voter_absence: bool,
    enable_candidate_absence: bool,
    seed: int | None,
):
    return run_monte_carlo(
        model=MODEL,
        selected_candidates=list(selected_candidates),
        iterations=iterations,
        voter_absence_probability=voter_absence_probability,
        volatility_level=volatility_level,
        candidate_absence_probability=candidate_absence_probability,
        enable_voter_absence=enable_voter_absence,
        enable_candidate_absence=enable_candidate_absence,
        seed=seed,
    )


@st.cache_data(show_spinner=False)
def get_single_run_result(
    selected_candidates: tuple[str, str, str],
    voter_absence_probability: float,
    volatility_level: int,
    candidate_absence_probability: float,
    enable_voter_absence: bool,
    enable_candidate_absence: bool,
    seed: int | None,
):
    return simulate_single_election(
        model=MODEL,
        selected_candidates=list(selected_candidates),
        voter_absence_probability=voter_absence_probability,
        volatility_level=volatility_level,
        candidate_absence_probability=candidate_absence_probability,
        enable_voter_absence=enable_voter_absence,
        enable_candidate_absence=enable_candidate_absence,
        rng=None if seed is None else random.Random(seed + 1),
    )


def _candidate_options_for_display(candidate_options: list[str]) -> list[str]:
    ordered = list(candidate_options)
    target_index = next((i for i, name in enumerate(ordered) if "მელქისედეკ" in name), None)
    if target_index is None:
        return ordered
    new_index = max(0, target_index - 3)
    if new_index == target_index:
        return ordered
    melqisedek = ordered.pop(target_index)
    ordered.insert(new_index, melqisedek)
    return ordered


def _normalize_selection(selection: list[str]) -> list[str]:
    if len(selection) != 3:
        raise ValueError("საჭიროა ზუსტად 3 კანდიდატი.")
    if len(set(selection)) != 3:
        raise ValueError("სამივე კანდიდატი განსხვავებული უნდა იყოს.")
    return selection


def _probability_df(monte_carlo_result: dict) -> pd.DataFrame:
    labels = [
        monte_carlo_result["selected_candidates"][0],
        monte_carlo_result["selected_candidates"][1],
        monte_carlo_result["selected_candidates"][2],
        RUNOFF_LABEL,
    ]
    return pd.DataFrame(
        {
            "შედეგი": labels,
            "ალბათობა": [
                monte_carlo_result["winner_probabilities"].get(label, 0.0)
                for label in labels
            ],
        }
    )


def _vote_stats_df(monte_carlo_result: dict) -> pd.DataFrame:
    rows = []
    for candidate, stats in monte_carlo_result["vote_statistics"].items():
        rows.append(
            {
                "კანდიდატი": candidate,
                "საშუალო ხმები": round(stats["average_votes"], 3),
                "მინიმუმი": stats["min_votes"],
                "მაქსიმუმი": stats["max_votes"],
            }
        )
    return pd.DataFrame(rows)


def _single_run_votes_df(single_run_result: dict, selected_candidates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "კანდიდატი": selected_candidates,
            "ხმები": [single_run_result["vote_totals"].get(candidate, 0) for candidate in selected_candidates],
        }
    )


def _render_static_vote_chart(vote_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(vote_df["კანდიდატი"], vote_df["ხმები"], color=BAR_COLORS[: len(vote_df)], width=0.6)
    ax.set_ylim(0, MAX_VOTERS)
    ax.set_ylabel("ხმები", fontweight="bold")
    ax.set_xlabel("")
    ax.set_title("ხმების განაწილება", fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_fontweight("bold")
        label.set_fontsize(10)
    for bar, value in zip(bars, vote_df["ხმები"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{int(value)}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_probability_chart(probability_df: pd.DataFrame) -> None:
    chart_df = probability_df.copy()
    chart_df["პროცენტი"] = chart_df["ალბათობა"] * 100
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    colors = BAR_COLORS + ["#7f7f7f"]
    bars = ax.bar(chart_df["შედეგი"], chart_df["პროცენტი"], color=colors[: len(chart_df)], width=0.6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("მოგების ალბათობა (%)", fontweight="bold")
    ax.set_xlabel("")
    ax.set_title("Monte Carlo-ს შედეგები", fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_fontweight("bold")
        label.set_fontsize(10)
    for bar, value in zip(bars, chart_df["პროცენტი"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _percentage_label(value: int) -> str:
    return f"{value}%"


def _display_probability_metrics(monte_carlo_result: dict) -> None:
    left, middle, right, far_right = st.columns(4)
    with left:
        st.metric(monte_carlo_result["selected_candidates"][0], f'{monte_carlo_result["winner_probabilities"][monte_carlo_result["selected_candidates"][0]]:.1%}')
    with middle:
        st.metric(monte_carlo_result["selected_candidates"][1], f'{monte_carlo_result["winner_probabilities"][monte_carlo_result["selected_candidates"][1]]:.1%}')
    with right:
        st.metric(monte_carlo_result["selected_candidates"][2], f'{monte_carlo_result["winner_probabilities"][monte_carlo_result["selected_candidates"][2]]:.1%}')
    with far_right:
        st.metric(RUNOFF_LABEL, f'{monte_carlo_result["winner_probabilities"][RUNOFF_LABEL]:.1%}')


def main() -> None:
    default_candidates = MODEL.default_parameters["default_selected_candidates"]
    default_params = MODEL.default_parameters
    display_candidate_options = _candidate_options_for_display(MODEL.candidate_options)

    st.title("საპატრიარქო არჩევნების სიმულაცია")
    st.caption("Excel-მოდელის Python/Streamlit რეკონსტრუქცია მაქსიმალურად ახლო ლოგიკით.")

    defaults = {
        "candidate_1": default_candidates[0],
        "candidate_2": default_candidates[1],
        "candidate_3": default_candidates[2],
        "iterations": int(default_params["default_iterations"]),
        "volatility_level": int(default_params["volatility_level"]),
        "voter_absence_percent": int(round(float(default_params["voter_absence_probability"]) * 100)),
        "candidate_absence_percent": int(round(float(default_params["candidate_absence_probability"]) * 100)),
        "enable_voter_absence": bool(default_params["enable_voter_absence"]),
        "enable_candidate_absence": bool(default_params["enable_candidate_absence"]),
        "show_charts": True,
        "show_full_single_run": False,
        "show_preferences_matrix": False,
        "seed_text": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    with st.sidebar:
        with st.form("simulation_form", clear_on_submit=False):
            st.header("პარამეტრები")

            candidate_1 = st.selectbox(
                "კანდიდატი 1",
                options=display_candidate_options,
                index=display_candidate_options.index(st.session_state["candidate_1"]),
            )

            candidate_2_options = [name for name in display_candidate_options if name != candidate_1]
            candidate_2_default = st.session_state.get("candidate_2", default_candidates[1])
            if candidate_2_default not in candidate_2_options:
                candidate_2_default = default_candidates[1] if default_candidates[1] in candidate_2_options else candidate_2_options[0]
            candidate_2 = st.selectbox(
                "კანდიდატი 2",
                options=candidate_2_options,
                index=candidate_2_options.index(candidate_2_default),
            )

            candidate_3_options = [name for name in display_candidate_options if name not in {candidate_1, candidate_2}]
            candidate_3_default = st.session_state.get("candidate_3", default_candidates[2])
            if candidate_3_default not in candidate_3_options:
                candidate_3_default = default_candidates[2] if default_candidates[2] in candidate_3_options else candidate_3_options[0]
            candidate_3 = st.selectbox(
                "კანდიდატი 3",
                options=candidate_3_options,
                index=candidate_3_options.index(candidate_3_default),
            )

            iterations = st.number_input("Monte Carlo ინტერაციები", min_value=1, max_value=100_000, value=st.session_state["iterations"], step=100)
            volatility_level = st.number_input("რყევადობის დონე", min_value=0, max_value=20, value=st.session_state["volatility_level"], step=1)
            voter_absence_percent = st.slider("ამომრჩევლის გაცდენის შანსი", min_value=0, max_value=100, value=st.session_state["voter_absence_percent"], step=1, format="%d%%")
            candidate_absence_percent = st.slider("კანდიდატის გაცდენის შანსი", min_value=0, max_value=100, value=st.session_state["candidate_absence_percent"], step=1, format="%d%%")

            st.caption(
                f"ამომრჩევლის გაცდენის შანსი: **{_percentage_label(voter_absence_percent)}**  \n"
                f"კანდიდატის გაცდენის შანსი: **{_percentage_label(candidate_absence_percent)}**"
            )

            enable_voter_absence = st.checkbox("გააქტიურდეს ამომრჩევლის გაცდენა", value=st.session_state["enable_voter_absence"])
            enable_candidate_absence = st.checkbox("გააქტიურდეს კანდიდატის გაცდენა", value=st.session_state["enable_candidate_absence"])
            show_charts = st.checkbox("გრაფიკების ჩვენება", value=st.session_state["show_charts"])
            show_full_single_run = st.checkbox("ერთი გაშვების სრული ცხრილის ჩვენება", value=st.session_state["show_full_single_run"])
            show_preferences_matrix = st.checkbox("პრეფერენციების მატრიცის ჩვენება", value=st.session_state["show_preferences_matrix"])
            seed_text = st.text_input("Seed (არასავალდებულო)", value=st.session_state["seed_text"])
            run_button = st.form_submit_button("სიმულაციის გაშვება", type="primary", use_container_width=True)

    if "last_single_run" not in st.session_state:
        st.session_state["last_single_run"] = None
    if "last_monte_carlo" not in st.session_state:
        st.session_state["last_monte_carlo"] = None
    if "last_parameters" not in st.session_state:
        st.session_state["last_parameters"] = None

    if run_button:
        st.session_state.update(
            {
                "candidate_1": candidate_1,
                "candidate_2": candidate_2,
                "candidate_3": candidate_3,
                "iterations": int(iterations),
                "volatility_level": int(volatility_level),
                "voter_absence_percent": int(voter_absence_percent),
                "candidate_absence_percent": int(candidate_absence_percent),
                "enable_voter_absence": enable_voter_absence,
                "enable_candidate_absence": enable_candidate_absence,
                "show_charts": show_charts,
                "show_full_single_run": show_full_single_run,
                "show_preferences_matrix": show_preferences_matrix,
                "seed_text": seed_text,
            }
        )

        seed = int(seed_text) if seed_text.strip() else None
        voter_absence_probability = voter_absence_percent / 100.0
        candidate_absence_probability = candidate_absence_percent / 100.0
        selected_candidates = _normalize_selection([candidate_1, candidate_2, candidate_3])
        selected_candidates_tuple = tuple(selected_candidates)

        with st.spinner("სიმულაცია მიმდინარეობს..."):
            monte_carlo_result = get_monte_carlo_result(
                selected_candidates=selected_candidates_tuple,
                iterations=int(iterations),
                voter_absence_probability=float(voter_absence_probability),
                volatility_level=int(volatility_level),
                candidate_absence_probability=float(candidate_absence_probability),
                enable_voter_absence=enable_voter_absence,
                enable_candidate_absence=enable_candidate_absence,
                seed=seed,
            )
            single_run_result = get_single_run_result(
                selected_candidates=selected_candidates_tuple,
                voter_absence_probability=float(voter_absence_probability),
                volatility_level=int(volatility_level),
                candidate_absence_probability=float(candidate_absence_probability),
                enable_voter_absence=enable_voter_absence,
                enable_candidate_absence=enable_candidate_absence,
                seed=seed,
            )

        st.session_state["last_single_run"] = single_run_result
        st.session_state["last_monte_carlo"] = monte_carlo_result
        st.session_state["last_parameters"] = {
            "selected_candidates": selected_candidates,
            "iterations": int(iterations),
            "volatility_level": int(volatility_level),
            "voter_absence_probability": float(voter_absence_probability),
            "candidate_absence_probability": float(candidate_absence_probability),
            "enable_voter_absence": enable_voter_absence,
            "enable_candidate_absence": enable_candidate_absence,
            "seed": seed,
        }

    mc = st.session_state["last_monte_carlo"]
    single = st.session_state["last_single_run"]

    if mc is None or single is None:
        st.info("აპი ახლა ფორმით მუშაობს: პარამეტრების ცვლილება თავისით აღარ გაუშვებს rerun-ს ყოველ ნაბიჯზე. აირჩიე პარამეტრები და დააჭირე **სიმულაციის გაშვება**.")
        return

    probability_df = _probability_df(mc)
    vote_stats_df = _vote_stats_df(mc)
    single_vote_totals_df = _single_run_votes_df(single, mc["selected_candidates"])

    st.subheader("ერთი კონკრეტული გაშვების დეტალები")
    winner_name = single["winner"]
    if winner_name == RUNOFF_LABEL:
        st.warning(f"### 🎰 შედეგი: **{winner_name}**", icon="⚠️")
    else:
        st.success(f"### 🏆 გამარჯვებული: **{winner_name}**", icon="🎉")

    st.caption(f"დასწრება: **{single['present_count']}** | გაცდენა: **{single['absent_count']}**")

    single_left, single_right = st.columns([0.85, 1.35])
    with single_left:
        st.table(single_vote_totals_df.set_index("კანდიდატი"))
    with single_right:
        if st.session_state["show_charts"]:
            _render_static_vote_chart(single_vote_totals_df)

    if st.session_state["show_full_single_run"]:
        with st.expander("ერთი გაშვების სრული ცხრილი", expanded=False):
            details_df = pd.DataFrame(single["rows"])
            st.dataframe(details_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("მოგების ალბათობები")
    _display_probability_metrics(mc)

    col1, col2 = st.columns([1.1, 0.95])
    with col1:
        if st.session_state["show_charts"]:
            _render_probability_chart(probability_df)
        st.table(
            probability_df.assign(ალბათობა=probability_df["ალბათობა"].map(lambda x: f"{x:.3%}"))
            .set_index("შედეგი")
        )

    with col2:
        st.subheader("საშუალო ხმები Monte Carlo-ზე")
        st.table(vote_stats_df.set_index("კანდიდატი"))
        st.info(
            f"საშუალოდ ესწრება {mc['average_present_count']:.2f} მღვდელმთავარი, ხოლო აცდენს {mc['average_absent_count']:.2f}."
        )

    if st.session_state["show_preferences_matrix"]:
        with st.expander("პრეფერენციების მატრიცა (Excel-ის N:V ბლოკის ანალოგი)", expanded=False):
            matrix_df = pd.DataFrame.from_dict(MODEL.preferences, orient="index")
            matrix_df.index.name = "მღვდელმთავარი"
            st.dataframe(matrix_df.reset_index(), use_container_width=True, hide_index=True)

    with st.expander("ლოგიკის შენიშვნები"):
        st.markdown(
            """
            - თითოეული ამომრჩევლისთვის სამი კანდიდატის ქულა ითვლება საბაზო პრეფერენციიდან და შემთხვევითი რყევიდან.
            - ქულის რყევა ზუსტად Excel-ის `RANDBETWEEN(-H7,H7)` ლოგიკით არის მოდელირებული.
            - ერთი ამომრჩევლის დონეზე თანაბარი ქულების შემთხვევაში უპირატესობა ენიჭება მარცხნიდან პირველ კანდიდატს.
            - საერთო გამარჯვებული მხოლოდ მაშინ ფიქსირდება, როცა ყველაზე მაღალი ხმა არ არის თანაბარი სხვასთან და 50%-ს აჭარბებს.
            - სხვა ყველა შემთხვევა ითვლება როგორც `მეორე ტური`.
            """
        )


if __name__ == "__main__":
    main()
