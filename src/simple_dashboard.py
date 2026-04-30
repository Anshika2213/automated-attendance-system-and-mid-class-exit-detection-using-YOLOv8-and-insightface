import streamlit as st
import sqlite3
import pandas as pd
import os
import time
from datetime import datetime

st.set_page_config(page_title="VIGIL-CLASS Dashboard", page_icon="V", layout="wide")

# DB path: works whether you run from src/ or project root
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DB_PATH        = os.path.join(_PROJECT_ROOT, 'data', 'database', 'attendance.db')
LIVE_FEED_PATH = os.path.join(_PROJECT_ROOT, 'data', 'live_feed.jpg')
if not os.path.exists(DB_PATH):
    DB_PATH        = os.path.join('data', 'database', 'attendance.db')
    LIVE_FEED_PATH = os.path.join('data', 'live_feed.jpg')


def get_conn():
    return sqlite3.connect(DB_PATH)

def get_today():
    return datetime.now().date().isoformat()

def get_current_period():
    conn = get_conn()
    now  = datetime.now().strftime('%H:%M')
    cur  = conn.cursor()
    cur.execute('''SELECT period_number, subject_name, start_time, end_time
                   FROM period_schedule
                   WHERE start_time <= ? AND end_time >= ?''', (now, now))
    row = cur.fetchone()
    conn.close()
    return row

def get_total_students():
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM students')
    n = cur.fetchone()[0]
    conn.close()
    return n

def get_attendance_stats(period=None):
    conn   = get_conn()
    today  = get_today()
    params = [today]
    where  = "WHERE date = ?"
    if period:
        where += " AND period = ?"
        params.append(period)
    cur = conn.cursor()
    cur.execute(f"SELECT status, COUNT(*) FROM attendance {where} GROUP BY status", params)
    stats = {'PRESENT': 0, 'TEMP_OUT': 0, 'BUNKED': 0, 'ABSENT': 0}
    for s, c in cur.fetchall():
        if s in stats:
            stats[s] = c
    conn.close()
    return stats

def get_schedule():
    conn = get_conn()
    df   = pd.read_sql_query(
        'SELECT period_number, subject_name, start_time, end_time '
        'FROM period_schedule ORDER BY period_number', conn)
    conn.close()
    return df

def get_student_list(period=None):
    """
    ONE ROW PER STUDENT — no duplicates.
    period=None  -> worst status across all periods today
    period=N     -> status for that specific period only
    """
    conn  = get_conn()
    today = get_today()

    if period:
        query = '''
            SELECT s.roll_no, s.name,
                   COALESCE(a.status,            'NOT YET') AS status,
                   COALESCE(a.entry_time,         '-')       AS entry_time,
                   COALESCE(a.exit_time,          '-')       AS exit_time,
                   COALESCE(a.duration_minutes,    0)        AS duration
            FROM students s
            LEFT JOIN attendance a
                ON s.roll_no = a.roll_no AND a.date = ? AND a.period = ?
            ORDER BY s.roll_no
        '''
        df = pd.read_sql_query(query, conn, params=[today, period])
    else:
        # Collapse multiple period rows into one using priority ranking:
        # BUNKED=4  ABSENT=3  TEMP_OUT=2  PRESENT=1  no record=0
        query = '''
            SELECT
                s.roll_no,
                s.name,
                CASE MAX(CASE a.status
                         WHEN 'BUNKED'   THEN 4
                         WHEN 'ABSENT'   THEN 3
                         WHEN 'TEMP_OUT' THEN 2
                         WHEN 'PRESENT'  THEN 1
                         ELSE 0 END)
                    WHEN 4 THEN 'BUNKED'
                    WHEN 3 THEN 'ABSENT'
                    WHEN 2 THEN 'TEMP_OUT'
                    WHEN 1 THEN 'PRESENT'
                    ELSE 'NOT YET'
                END AS status,
                COALESCE(MIN(a.entry_time),        '-') AS entry_time,
                COALESCE(MAX(a.exit_time),         '-') AS exit_time,
                COALESCE(SUM(a.duration_minutes),   0)  AS duration
            FROM students s
            LEFT JOIN attendance a ON s.roll_no = a.roll_no AND a.date = ?
            GROUP BY s.roll_no, s.name
            ORDER BY s.roll_no
        '''
        df = pd.read_sql_query(query, conn, params=[today])

    conn.close()
    return df

def get_recent_events(limit=10):
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute('''
        SELECT s.name, a.roll_no, a.status, a.entry_time, a.exit_time, a.period
        FROM attendance a
        JOIN students s ON a.roll_no = s.roll_no
        WHERE a.date = ?
        ORDER BY COALESCE(a.exit_time, a.entry_time) DESC
        LIMIT ?
    ''', [get_today(), limit])
    rows = cur.fetchall()
    conn.close()
    return rows

def get_low_attendance(threshold=75):
    conn  = get_conn()
    query = '''
        SELECT s.roll_no, s.name,
               COUNT(CASE WHEN a.status = 'PRESENT' THEN 1 END) AS present_count,
               COUNT(a.id)                                       AS total_periods,
               ROUND(100.0 * COUNT(CASE WHEN a.status = 'PRESENT' THEN 1 END)
                     / NULLIF(COUNT(a.id), 0), 1)               AS attendance_pct
        FROM students s
        LEFT JOIN attendance a ON s.roll_no = a.roll_no
        GROUP BY s.roll_no, s.name
        HAVING total_periods > 0
        ORDER BY attendance_pct ASC
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df[df['attendance_pct'] < threshold] if not df.empty else df

STATUS_ICON  = {'PRESENT': '🟢', 'TEMP_OUT': '🟡', 'BUNKED': '🔴', 'ABSENT': '⚫', 'NOT YET': '⬜'}
STATUS_COLOR = {'PRESENT': '#1a9e3f', 'TEMP_OUT': '#e69c00', 'BUNKED': '#d93025',
                'ABSENT': '#5f5f5f', 'NOT YET': '#888888'}


# Guard
if not os.path.exists(DB_PATH):
    st.error(f"Database not found: {DB_PATH}")
    st.info("Run `python src/database.py` first.")
    st.stop()


# CSS
st.markdown("""
<style>
.metric-card{background:#1e2130;border-radius:12px;padding:20px 24px;
             text-align:center;border:1px solid #2d3149;}
.metric-value{font-size:2.6rem;font-weight:700;margin:0;}
.metric-label{font-size:.82rem;color:#aaa;margin-top:4px;
              text-transform:uppercase;letter-spacing:.05em;}
.period-banner{background:#1a3a5c;border-left:4px solid #3a8fd4;
               border-radius:0 8px 8px 0;padding:12px 18px;margin-bottom:14px;}
.alert-box{background:#2d1a1a;border-left:4px solid #d93025;
           border-radius:0 8px 8px 0;padding:10px 16px;margin-bottom:6px;}
.event-row{background:#1a1d2e;border-radius:6px;
           padding:7px 12px;margin-bottom:5px;font-size:.86rem;}
.sec-hdr{font-size:.9rem;font-weight:600;color:#ccc;text-transform:uppercase;
         letter-spacing:.07em;margin-bottom:8px;padding-bottom:5px;
         border-bottom:1px solid #2d3149;}
</style>
""", unsafe_allow_html=True)


# Header
h1, h2 = st.columns([3, 1])
with h1:
    st.markdown("## VIGIL-CLASS - Attendance Dashboard")
with h2:
    st.markdown(
        f"<div style='text-align:right;color:#666;padding-top:14px;'>"
        f"{datetime.now().strftime('%d %b %Y  %H:%M:%S')}</div>",
        unsafe_allow_html=True)
st.divider()


# Current period banner
cp = get_current_period()
if cp:
    p_num, p_sub, p_start, p_end = cp
    st.markdown(
        f'<div class="period-banner">'
        f'<b style="font-size:1.05rem;">Period {p_num} - {p_sub}</b>'
        f'<span style="color:#aaa;margin-left:14px;">{p_start} to {p_end}</span>'
        f'<span style="background:#1a9e3f;color:#fff;border-radius:10px;'
        f'padding:2px 8px;font-size:.75rem;margin-left:10px;">LIVE</span>'
        f'</div>', unsafe_allow_html=True)
    stats = get_attendance_stats(period=p_num)
else:
    st.markdown(
        '<div style="background:#1e2130;border-radius:8px;padding:10px 16px;'
        'border-left:4px solid #444;margin-bottom:14px;color:#888;">'
        'No period active right now - system on standby</div>',
        unsafe_allow_html=True)
    stats = get_attendance_stats()


# Metric cards
total    = get_total_students()
present  = stats['PRESENT']
temp_out = stats['TEMP_OUT']
bunked   = stats['BUNKED']
absent   = stats['ABSENT']
pct      = round(100 * present / total, 1) if total > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
for col, val, label, color in [
    (c1, total,    "Total Students", "#3a8fd4"),
    (c2, present,  "Present",        "#1a9e3f"),
    (c3, temp_out, "Temp Out",       "#e69c00"),
    (c4, bunked,   "Bunked",         "#d93025"),
    (c5, f"{pct}%","Attendance %",   "#3a8fd4"),
]:
    col.markdown(
        f'<div class="metric-card">'
        f'<p class="metric-value" style="color:{color};">{val}</p>'
        f'<p class="metric-label">{label}</p>'
        f'</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# Bunking alerts
if bunked > 0:
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute('''
        SELECT s.name, a.roll_no, a.exit_time, a.period
        FROM attendance a JOIN students s ON a.roll_no = s.roll_no
        WHERE a.date = ? AND a.status = 'BUNKED'
        ORDER BY a.exit_time DESC
    ''', [get_today()])
    bl = cur.fetchall()
    conn.close()
    st.markdown('<p class="sec-hdr">Bunking Alerts</p>', unsafe_allow_html=True)
    for bname, broll, bexit, bperiod in bl:
        st.markdown(
            f'<div class="alert-box">🚨 <b>Roll-{broll} ({bname})</b>'
            f' - bunked Period {bperiod}'
            f' | Left at: {bexit or "unknown"}'
            f' | Did not return within 15 min</div>',
            unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


# Main layout: camera + activity | student table
left_col, right_col = st.columns([1.2, 1.8])

with left_col:
    st.markdown('<p class="sec-hdr">Live Camera Feed</p>', unsafe_allow_html=True)
    if os.path.exists(LIVE_FEED_PATH):
        st.image(LIVE_FEED_PATH, use_container_width=True,
                 caption="Door Camera - updates every 5s")
    else:
        st.markdown(
            '<div style="background:#1e2130;border-radius:8px;padding:40px;'
            'text-align:center;color:#555;border:2px dashed #333;">'
            'Camera feed not available<br>'
            '<small>Start tracking to see live feed</small></div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="sec-hdr">Recent Activity</p>', unsafe_allow_html=True)
    events = get_recent_events(limit=10)
    if events:
        for ename, eroll, estatus, eentry, eexit, eperiod in events:
            icon  = STATUS_ICON.get(estatus, '⬜')
            color = STATUS_COLOR.get(estatus, '#888')
            if estatus == 'PRESENT':
                detail = f"entered at {eentry}"
            elif estatus == 'TEMP_OUT':
                detail = f"left at {eexit}"
            elif estatus == 'BUNKED':
                detail = f"bunked - left at {eexit}"
            elif estatus == 'ABSENT':
                detail = "never entered"
            else:
                detail = ""
            st.markdown(
                f'<div class="event-row">{icon} <b>{ename}</b> (Roll-{eroll})'
                f' - <span style="color:{color};font-weight:600;">{estatus}</span>'
                f' - P{eperiod} {detail}</div>',
                unsafe_allow_html=True)
    else:
        st.info("No activity yet today.")


with right_col:
    st.markdown('<p class="sec-hdr">Student Status</p>', unsafe_allow_html=True)

    sched_df = get_schedule()
    options  = ['All Periods'] + [
        f"Period {r['period_number']} - {r['subject_name']}"
        for _, r in sched_df.iterrows()
    ]
    selected      = st.selectbox("Filter:", options, label_visibility='collapsed')
    filter_period = None if selected == 'All Periods' else int(selected.split(' ')[1])

    df = get_student_list(period=filter_period)

    if df.empty:
        st.info("No student data yet.")
    else:
        df['Status'] = df['status'].map(lambda s: f"{STATUS_ICON.get(s,'⬜')} {s}")
        disp = df[['roll_no', 'name', 'Status', 'entry_time', 'exit_time', 'duration']].copy()
        disp.columns = ['Roll No', 'Name', 'Status', 'Entry', 'Exit', 'Min in Class']

        def _style(val):
            for s, c in STATUS_COLOR.items():
                if s in str(val):
                    return f'color:{c};font-weight:bold;'
            return ''

        st.dataframe(
            disp.style.applymap(_style, subset=['Status']),
            use_container_width=True, height=430, hide_index=True)

        csv = df[['roll_no', 'name', 'status', 'entry_time',
                  'exit_time', 'duration']].to_csv(index=False)
        st.download_button(
            label="Download Report (CSV)", data=csv,
            file_name=f"attendance_{get_today()}_p{filter_period or 'all'}.csv",
            mime='text/csv', use_container_width=True)


# Schedule + low attendance
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
bl_col, br_col = st.columns(2)

with bl_col:
    st.markdown('<p class="sec-hdr">Period Schedule</p>', unsafe_allow_html=True)
    now_str = datetime.now().strftime('%H:%M')
    rows_html = ""
    for _, row in sched_df.iterrows():
        active = row['start_time'] <= now_str <= row['end_time']
        bg     = "#1a3a5c" if active else "#1a1d2e"
        badge  = (' <span style="background:#1a9e3f;color:#fff;border-radius:8px;'
                  'padding:1px 7px;font-size:.7rem;">NOW</span>') if active else ""
        rows_html += (
            f'<div style="background:{bg};border-radius:6px;padding:8px 12px;'
            f'margin-bottom:4px;font-size:.86rem;">'
            f'<b>P{row["period_number"]}</b> &nbsp;'
            f'{row["start_time"]} to {row["end_time"]} &nbsp;|&nbsp; '
            f'{row["subject_name"]}{badge}</div>')
    st.markdown(rows_html, unsafe_allow_html=True)

with br_col:
    st.markdown('<p class="sec-hdr">Low Attendance Warning (below 75%)</p>',
                unsafe_allow_html=True)
    low_df = get_low_attendance(threshold=75)
    if low_df.empty:
        st.success("All students above 75% attendance!")
    else:
        for _, row in low_df.iterrows():
            pv    = row['attendance_pct']
            color = "#d93025" if pv < 50 else "#e69c00"
            st.markdown(
                f'<div style="background:#1e2130;border-radius:6px;padding:8px 12px;'
                f'margin-bottom:4px;font-size:.86rem;border-left:3px solid {color};">'
                f'<b>Roll-{int(row["roll_no"])}</b> - {row["name"]}'
                f' &nbsp;<span style="color:{color};font-weight:700;">{pv}%</span>'
                f' ({int(row["present_count"])}/{int(row["total_periods"])} periods)'
                f'</div>', unsafe_allow_html=True)


# Footer + auto-refresh
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center;color:#333;font-size:.72rem;'>"
    "Auto-refreshes every 5 seconds | VIGIL-CLASS v1.0</div>",
    unsafe_allow_html=True)

time.sleep(5)
st.rerun()