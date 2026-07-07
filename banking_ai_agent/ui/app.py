"""
ui/app.py
----------
Streamlit dashboard for the Banking Customer Support Multi-Agent system.

Run with:
    streamlit run ui/app.py

Tabs (mapped directly to the brief's Part 2 requirements):
  1. Live Agent          - accept user input, simulate agent routing, display
                           classification/response/DB interaction, and give a
                           thumbs up/down feedback loop on each response.
  2. Tickets             - browse / update the support_tickets table.
  3. Test Scenarios      - exercise each agent role individually and directly
                           (Classifier, Feedback Handler, Query Handler),
                           without going through the full pipeline.
  4. Evaluation          - run the QA-based scoring suite, see classifier
                           test-case coverage, and the live routing success
                           rate.
  5. Logs & Debugging    - prompt traces, classification output, ticket
                           actions, agent success/failure rate, and the
                           flagged-for-review queue fed by the feedback loop.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
from database import db
from orchestrator import Orchestrator
from agents.classifier_agent import ClassifierAgent
from agents.feedback_agent import FeedbackAgent
from agents.query_agent import QueryAgent
import evaluation

st.set_page_config(page_title="Banking Support AI Agent", page_icon="🏦", layout="wide")

db.init_db()

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = Orchestrator()
if "history" not in st.session_state:
    st.session_state.history = []
if "eval_report" not in st.session_state:
    st.session_state.eval_report = None

st.title("🏦 Banking Customer Support AI Agent")
st.caption("Multi-agent architecture: Classifier → Feedback Handler / Query Handler")

tab_agent, tab_tickets, tab_scenarios, tab_eval, tab_logs = st.tabs(
    ["💬 Live Agent", "🎫 Tickets", "🧭 Test Scenarios", "🧪 Evaluation", "🪵 Logs & Debugging"]
)

# ----------------------------------------------------------------------
# Tab 1: Live Agent
# ----------------------------------------------------------------------
with tab_agent:
    col_input, col_examples = st.columns([2, 1])

    with col_examples:
        st.markdown("**Try a sample message:**")
        samples = [
            "Thanks for sorting out my net banking login issue.",
            "My debit card replacement still hasn't arrived.",
            "Could you check the status of ticket 650932?",
        ]
        for s in samples:
            if st.button(s, key=s):
                st.session_state["prefill"] = s

    with col_input:
        customer_name = st.text_input("Customer name", value="Aditi")
        user_message = st.text_area(
            "Customer message",
            value=st.session_state.get("prefill", ""),
            placeholder="e.g. My debit card replacement still hasn't arrived.",
            height=100,
        )

        if st.button("Send", type="primary"):
            if user_message.strip():
                result = st.session_state.orchestrator.handle_message(user_message, customer_name)
                st.session_state.history.insert(0, {
                    "log_id": result.log_id,
                    "message": user_message,
                    "customer": customer_name,
                    "classification": result.classification,
                    "agent_path": result.agent_path,
                    "response": result.response,
                    "ticket_id": result.ticket_id,
                    "used_fallback": result.used_fallback,
                    "rated": None,
                })
                st.session_state["prefill"] = ""

    st.divider()
    st.subheader("Conversation trace")
    for i, turn in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.write(f"**{turn['customer']}:** {turn['message']}")
        with st.chat_message("assistant"):
            badge = {
                "POSITIVE_FEEDBACK": "🟢 Positive Feedback",
                "NEGATIVE_FEEDBACK": "🔴 Negative Feedback",
                "QUERY": "🔵 Query",
            }[turn["classification"]]
            st.markdown(f"`{badge}` — routed via **{turn['agent_path']}**")
            if turn["used_fallback"]:
                st.warning("Classifier fell back to a default label - flagged as a routing failure in Logs.")
            st.write(turn["response"])
            if turn["ticket_id"]:
                st.info(f"Ticket #{turn['ticket_id']} created")

            # --- Feedback loop: this is the "agent improvement loop" the
            # brief's Logs & Debugging section calls out as optional. ---
            fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])
            if turn["rated"] is None:
                if fb_col1.button("👍", key=f"up_{i}_{turn['log_id']}"):
                    db.log_feedback(turn["log_id"], "up")
                    st.session_state.history[i]["rated"] = "up"
                    st.rerun()
                if fb_col2.button("👎", key=f"down_{i}_{turn['log_id']}"):
                    db.log_feedback(turn["log_id"], "down")
                    st.session_state.history[i]["rated"] = "down"
                    st.rerun()
            else:
                fb_col1.caption(f"Rated {'👍' if turn['rated'] == 'up' else '👎'}")

# ----------------------------------------------------------------------
# Tab 2: Tickets
# ----------------------------------------------------------------------
with tab_tickets:
    st.subheader("Support tickets")
    tickets = db.list_tickets()
    if tickets:
        st.dataframe(tickets, use_container_width=True)
        ticket_ids = [t["ticket_id"] for t in tickets]
        selected = st.selectbox("Update a ticket's status", ticket_ids)
        new_status = st.selectbox("New status", ["Open", "In Progress", "Resolved"])
        if st.button("Update status"):
            db.update_ticket_status(selected, new_status)
            st.success(f"Ticket #{selected} updated to {new_status}")
            st.rerun()
    else:
        st.info("No tickets yet. Submit negative feedback in the Live Agent tab to create one.")

# ----------------------------------------------------------------------
# Tab 3: Test Scenarios (per-agent role, bypassing the full pipeline)
# ----------------------------------------------------------------------
with tab_scenarios:
    st.subheader("Exercise a single agent role directly")
    st.caption("Useful for isolating a bug: does the Classifier mislabel the message, "
               "or does a downstream handler misbehave given the correct label?")

    role = st.radio("Agent role", ["Classifier Agent", "Feedback Handler Agent", "Query Handler Agent"], horizontal=True)

    if role == "Classifier Agent":
        msg = st.text_area("Message to classify", "Thanks for resolving my issue quickly!")
        if st.button("Run Classifier"):
            label, used_fallback = ClassifierAgent().classify(msg)
            st.write(f"**Predicted label:** `{label}`")
            if used_fallback:
                st.warning("Model output didn't match a valid label - defaulted to QUERY.")

    elif role == "Feedback Handler Agent":
        branch = st.selectbox("Branch", ["Positive", "Negative"])
        name = st.text_input("Customer name", "Aditi")
        msg = st.text_area(
            "Feedback message",
            "Thanks for resolving my issue!" if branch == "Positive" else "My card replacement never arrived.",
        )
        if st.button("Run Feedback Handler"):
            fa = FeedbackAgent()
            if branch == "Positive":
                st.write(fa.handle_positive(msg, name))
            else:
                response, ticket_id = fa.handle_negative(msg, name)
                st.write(response)
                st.info(f"Ticket #{ticket_id} created in support_tickets")

    else:  # Query Handler Agent
        msg = st.text_area("Query message", "Could you check the status of ticket 650932?")
        if st.button("Run Query Handler"):
            qa = QueryAgent()
            extracted = qa.extract_ticket_id(msg)
            st.write(f"**Extracted ticket ID:** `{extracted}`")
            st.write(qa.handle(msg))

# ----------------------------------------------------------------------
# Tab 4: Evaluation
# ----------------------------------------------------------------------
with tab_eval:
    st.subheader("Model evaluation")
    st.caption("QA-based scoring, classifier test-case coverage, and routing success rate.")

    if st.button("▶ Run evaluation suite", type="primary"):
        with st.spinner("Running classifier + response-quality evaluation..."):
            st.session_state.eval_report = evaluation.generate_evaluation_report()

    report = st.session_state.eval_report
    if report:
        clf = report["classifier_evaluation"]
        qa = report["response_quality_evaluation"]
        routing = report["routing_success_rate"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Classifier accuracy", f"{clf['overall_accuracy'] * 100:.0f}%",
                   help=f"{clf['correct']}/{clf['total_cases']} test cases")
        m2.metric("Response quality (QA rubric)", f"{qa['overall_quality_score'] * 100:.0f}%")
        m3.metric("Routing success rate", f"{routing['rate'] * 100:.0f}%",
                   help=f"{routing['successful']}/{routing['total']} logged interactions")

        st.markdown("#### Classifier accuracy by label (test case coverage)")
        st.bar_chart(clf["per_label_accuracy"])

        st.markdown("#### Classifier test cases")
        st.dataframe(clf["cases"], use_container_width=True)

        st.markdown("#### Response quality — rubric checks per case")
        st.dataframe(
            [{"message": c["message"], "classification": c["classification"],
              "pass_rate": c["pass_rate"], **c["checks"]} for c in qa["cases"]],
            use_container_width=True,
        )
    else:
        st.info("Click 'Run evaluation suite' to score the classifier and response quality "
                "against the labeled dataset in tests/eval_dataset.json.")

# ----------------------------------------------------------------------
# Tab 5: Logs & Debugging
# ----------------------------------------------------------------------
with tab_logs:
    st.subheader("Interaction logs")
    logs = db.list_logs()
    if logs:
        routing = db.routing_success_rate()
        fb = db.feedback_stats()

        m1, m2, m3 = st.columns(3)
        m1.metric("Routing success rate", f"{routing['rate'] * 100:.0f}%",
                   help=f"{routing['successful']}/{routing['total']} interactions")
        m2.metric("👍 Positive feedback", fb["overall"].get("up", 0))
        m3.metric("👎 Flagged for review", fb["overall"].get("down", 0))

        st.markdown("#### Full interaction log (includes prompt trace & success flag)")
        st.dataframe(logs, use_container_width=True)

        st.markdown("#### Routing distribution")
        by_class = {}
        for log in logs:
            by_class[log["classification"]] = by_class.get(log["classification"], 0) + 1
        st.bar_chart(by_class)

        if fb["flagged_for_review"]:
            st.markdown("#### 🚩 Flagged for review (agent improvement loop)")
            st.caption("Interactions the user marked 👎 - review these to spot systematic response-quality issues.")
            st.dataframe(fb["flagged_for_review"], use_container_width=True)

        with st.expander("View a prompt trace for a specific log entry"):
            log_ids = [l["log_id"] for l in logs]
            chosen = st.selectbox("Log ID", log_ids)
            chosen_log = next(l for l in logs if l["log_id"] == chosen)
            st.code(chosen_log["prompt_trace"], language="text")
    else:
        st.info("No interactions logged yet. Try the Live Agent tab first.")
