# production_tracker.py
import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Database Setup
# -----------------------------
def init_db():
    conn = sqlite3.connect("production_tracker.db")
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Lines (
        line_id TEXT PRIMARY KEY,
        description TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Machines (
        machine_id INTEGER PRIMARY KEY AUTOINCREMENT,
        line_id TEXT,
        machine_name TEXT,
        machine_type TEXT,
        FOREIGN KEY(line_id) REFERENCES Lines(line_id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Operators (
        operator_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        assigned_line TEXT,
        FOREIGN KEY(assigned_line) REFERENCES Lines(line_id)
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ActivityLogs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        line_id TEXT,
        machine_id INT,
        operator_id INT,
        activity_type TEXT,
        part_description TEXT,
        start_time TEXT,
        end_time TEXT,
        duration_minutes FLOAT,
        notes TEXT,
        FOREIGN KEY(line_id) REFERENCES Lines(line_id),
        FOREIGN KEY(machine_id) REFERENCES Machines(machine_id),
        FOREIGN KEY(operator_id) REFERENCES Operators(operator_id)
    );
    """)

    # Predefined machine list
    machines = [
        ("Rougher Roller Turner", "Rougher"),
        ("Rougher Pump", "Rougher"),
        ("Rougher Body Grinder", "Rougher"),
        ("Rougher Marposs Gauging System", "Rougher"),
        ("Rougher Conveyor", "Rougher"),
        ("Semi-Finisher Pump", "Semi-Finisher"),
        ("Semi-Finisher Marposs Gauging System", "Semi-Finisher"),
        ("Semi-Finisher Body Grinder", "Semi-Finisher"),
        ("Semi-Finisher Conveyor", "Semi-Finisher"),
        ("SEG", "Semi-Finisher"),
        ("Finisher Body Grinder", "Finisher"),
        ("Finisher Pump", "Finisher"),
        ("Finisher Marposs Gauging System", "Finisher"),
        ("Finisher Conveyor", "Finisher"),
        ("Honer", "Finisher"),
        ("ECT", "Finisher"),
        ("IBG", "Finisher"),
        ("Vision System", "Inspection"),
        ("Sealer Unit", "Finisher")
    ]

    # Insert lines and machines
    for i in range(1, 23):  # R1 to R22
        line_id = f"R{i}"
        cursor.execute("INSERT OR IGNORE INTO Lines (line_id, description) VALUES (?, ?)",
                       (line_id, f"Production Line {i}"))
        for machine_name, machine_type in machines:
            cursor.execute("INSERT INTO Machines (line_id, machine_name, machine_type) VALUES (?, ?, ?)",
                           (line_id, machine_name, machine_type))

    conn.commit()
    conn.close()

# -----------------------------
# Streamlit App
# -----------------------------
def app():
    st.title("🏭 AI-Powered Production Tracker")

    conn = sqlite3.connect("production_tracker.db")
    cursor = conn.cursor()

    # Fetch lines and machines
    lines = pd.read_sql("SELECT * FROM Lines", conn)
    machines = pd.read_sql("SELECT * FROM Machines", conn)
    operators = pd.read_sql("SELECT * FROM Operators", conn)

    # Activity Logging Form
    st.subheader("Log Activity")
    line_id = st.selectbox("Select Line", lines["line_id"].tolist())
    part_description = st.text_input("Part Description")
    machine_name = st.selectbox("Select Machine", machines[machines["line_id"] == line_id]["machine_name"].tolist())
    operator_name = st.text_input("Operator Name")
    activity_type = st.selectbox("Activity Type", ["Setup", "Calibration", "Cleaning", "Trial Run", "Downtime"])
    start_time_str = st.text_input("Enter Start Time (HH:MM)", "08:00")
    end_time_str = st.text_input("Enter End Time (HH:MM)", "09:00")
    notes = st.text_area("Notes / Downtime Reason")

    if st.button("Submit Activity"):
        # Ensure operator exists
        cursor.execute("INSERT OR IGNORE INTO Operators (name, assigned_line) VALUES (?, ?)", (operator_name, line_id))
        conn.commit()

        # Get IDs
        machine_id = machines[(machines["line_id"] == line_id) & (machines["machine_name"] == machine_name)]["machine_id"].values[0]
        operator_id = pd.read_sql("SELECT operator_id FROM Operators WHERE name=? AND assigned_line=?", conn, params=(operator_name, line_id))["operator_id"].values[0]

        # Parse times and calculate duration
        try:
            start_time = datetime.strptime(start_time_str, "%H:%M")
            end_time = datetime.strptime(end_time_str, "%H:%M")
            duration = (end_time - start_time).seconds / 60

            # Insert log
            cursor.execute("""
            INSERT INTO ActivityLogs (line_id, machine_id, operator_id, activity_type, part_description, start_time, end_time, duration_minutes, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (line_id, machine_id, operator_id, activity_type, part_description,
                  start_time_str, end_time_str, duration, notes))
            conn.commit()
            st.success("✅ Activity logged successfully!")
        except ValueError:
            st.error("⚠️ Please enter time in HH:MM format (e.g., 08:30).")

    # Show Logs
    st.subheader("Activity Logs")
    logs = pd.read_sql("SELECT * FROM ActivityLogs", conn)
    st.dataframe(logs)

    # AI Insights
    st.subheader("AI Insights")
    if not logs.empty:
        # Encode categorical features
        le_line = LabelEncoder()
        le_activity = LabelEncoder()
        le_part = LabelEncoder()

        logs["line_enc"] = le_line.fit_transform(logs["line_id"])
        logs["activity_enc"] = le_activity.fit_transform(logs["activity_type"])
        logs["part_enc"] = le_part.fit_transform(logs["part_description"].fillna("Unknown"))

        X = logs[["line_enc", "activity_enc", "part_enc"]]
        y = logs["duration_minutes"]

        # Train predictive model
        model = RandomForestRegressor()
        model.fit(X, y)

        # Train anomaly detector
        anomaly_model = IsolationForest(contamination=0.1)
        anomaly_model.fit(X)

        # Prediction UI
        pred_line = st.selectbox("Predict for Line", lines["line_id"].tolist())
        pred_activity = st.selectbox("Predict for Activity", ["Setup", "Calibration", "Cleaning", "Trial Run", "Downtime"])
        pred_part = st.text_input("Predict for Part Description")

        if st.button("Predict Setup Time"):
            try:
                pred = model.predict([[le_line.transform([pred_line])[0],
                                       le_activity.transform([pred_activity])[0],
                                       le_part.transform([pred_part])[0] if pred_part in le_part.classes_ else 0]])
                st.info(f"Predicted setup time: {pred[0]:.2f} minutes")
            except Exception:
                st.error("⚠️ Please enter a valid part description seen in logs.")

        if st.button("Check Anomaly"):
            try:
                result = anomaly_model.predict([[le_line.transform([pred_line])[0],
                                                 le_activity.transform([pred_activity])[0],
                                                 le_part.transform([pred_part])[0] if pred_part in le_part.classes_ else 0]])
                if result[0] == -1:
                    st.warning("⚠️ This activity looks abnormal compared to history.")
                else:
                    st.success("✅ Activity is within normal range.")
            except Exception:
                st.error("⚠️ Unable to check anomaly for this input.")

    conn.close()

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    init_db()
    app()
