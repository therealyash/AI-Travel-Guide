import os
import time
import json
import requests
from datetime import datetime, timedelta

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


DB_FAISS_PATH = "vectorstore/db_faiss"
HF_TOKEN = os.getenv("HF_TOKEN")  # kept as-is
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_API_KEY")

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def load_llm(huggingface_repo_id):
    endpoint = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.4,
        model_kwargs={"max_length": 768}
    )
    return ChatHuggingFace(llm=endpoint)

def set_custom_prompt():
    CUSTOM_PROMPT_TEMPLATE = """
    You are an AI-powered travel planner and guide.
    Use only the information in the provided context (destination facts, attractions, seasons, logistics) to help plan trips.
    If the context does not contain enough info, say you don't know.

    Produce practical, concise output with:
    - Best time to visit (if available)
    - Day-wise itinerary (with time blocks if possible)
    - Key attractions/activities and nearby alternatives
    - Local tips (transport, food, passes) when present in context

    IMPORTANT:
    - Do NOT invent facts not in context.
    - Keep tone clear and actionable.

    Context: {context}
    Question: {question}
    """
    return ChatPromptTemplate.from_messages([
        ("system", CUSTOM_PROMPT_TEMPLATE),
        ("human", "{question}")
    ])

def get_qa_chain():
    vectorstore = get_vectorstore()
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(HUGGINGFACE_REPO_ID),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 4}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt()}
    )
    return qa_chain

def _fetch_json(url, params):
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def geocode_city(city: str, api_key: str):
    url = "http://api.openweathermap.org/geo/1.0/direct"
    data = _fetch_json(url, {"q": city, "limit": 1, "appid": api_key})
    if not data:
        return None
    return {"name": data[0]["name"], "lat": data[0]["lat"], "lon": data[0]["lon"], "country": data[0].get("country", "")}

def current_weather(lat: float, lon: float, api_key: str):
    url = "https://api.openweathermap.org/data/2.5/weather"
    return _fetch_json(url, {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"})

def forecast_5day(lat: float, lon: float, api_key: str):
    url = "https://api.openweathermap.org/data/2.5/forecast"
    return _fetch_json(url, {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"})

def render_weather_block(city: str, travel_dates: tuple[str, str] | None, api_key: str):
    if not city or api_key in (None, "", "YOUR_OPENWEATHER_API_KEY"):
        st.info("Add a valid OpenWeather API key in the sidebar to show weather.")
        return

    geo = geocode_city(city, api_key)
    if not geo:
        st.warning("Couldn't geocode the city.")
        return

    col1, col2 = st.columns(2)
    with col1:
        try:
            cw = current_weather(geo["lat"], geo["lon"], api_key)
            st.subheader("Current Weather")
            st.metric(
                label=f"{geo['name']}, {geo.get('country','')}",
                value=f"{cw['main']['temp']}°C",
                delta=f"Feels {cw['main']['feels_like']}°C"
            )
            st.caption(f"{cw['weather'][0]['main']}: {cw['weather'][0]['description'].title()}")
        except Exception as e:
            st.error(f"Current weather error: {e}")

    with col2:
        try:
            st.subheader("Forecast (Next 5 Days)")
            fc = forecast_5day(geo["lat"], geo["lon"], api_key)
            by_day = {}
            for item in fc["list"]:
                date = item["dt_txt"].split(" ")[0]
                hour = item["dt_txt"].split(" ")[1].split(":")[0]
                if date not in by_day or hour == "12":
                    by_day[date] = item
            for d, item in list(by_day.items())[:5]:
                temp = item["main"]["temp"]
                desc = item["weather"][0]["description"].title()
                st.write(f"**{d}** — {temp}°C, {desc}")
        except Exception as e:
            st.error(f"Forecast error: {e}")


def estimate_budget(destination: str, days: int, tier: str):
    region_cost = {
        "europe": 180,
        "usa": 200,
        "japan": 160,
        "australia": 170,
        "south asia": 70,
        "southeast asia": 90,
        "middle east": 120,
        "latin america": 100,
        "africa": 80,
        "default": 110
    }

    tier_factor = {
        "Backpacker": 0.6,
        "Mid-range": 1.0,
        "Luxury": 2.0,
        "-": 1.0
    }

    dest_lower = destination.lower()
    region = "default"
    for r in region_cost:
        if r in dest_lower:
            region = r
            break

    base_per_day = region_cost.get(region, region_cost["default"])
    total = base_per_day * days * tier_factor.get(tier, 1.0)
    low, high = int(total * 0.85), int(total * 1.15)

    return {
        "destination": destination,
        "days": days,
        "tier": tier,
        "region": region.title(),
        "estimate_range": f"${low:,} - ${high:,} USD",
        "avg_per_day": f"${int(total/days):,}/day"
    }

def render_budget_section(destination: str, date_from, date_to, tier: str):
    if not destination or not date_from or not date_to:
        return
    days = (date_to - date_from).days + 1
    budget_data = estimate_budget(destination, days, tier)

    st.subheader("💰 Estimated Budget")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"Total ({days} days, {tier})", budget_data["estimate_range"])
    with col2:
        st.metric("Average per day", budget_data["avg_per_day"])
    st.caption(f"Region detected: {budget_data['region']}")


def reasoning_prompt():
    tmpl = """
    Provide a SHORT, STRUCTURED JSON outline of how you plan this trip.
    This is not your hidden reasoning—just a concise trace.

    JSON schema:
    {{
      "inputs_considered": ["bullet", ...],
      "shortlist_destinations": ["City A", "City B", ...],
      "planning_steps": ["Step 1...", "Step 2...", ...],
      "final_structure": ["Day 1...", "Day 2...", ...]
    }}

    User Query: {q}
    """
    return ChatPromptTemplate.from_messages([
        ("system", tmpl),
        ("human", "{q}")
    ])


def generate_concise_trace(user_query: str) -> dict:
    llm = load_llm(HUGGINGFACE_REPO_ID)
    prompt = reasoning_prompt().format_messages(q=user_query)
    resp = llm.invoke(prompt)
    text = getattr(resp, "content", str(resp))
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {
            "inputs_considered": ["Destination", "Dates", "Budget", "Interests", "Season"],
            "shortlist_destinations": [],
            "planning_steps": ["Collect inputs", "Match with context", "Assemble plan"],
            "final_structure": ["Day 1: City tour", "Day 2: Nature", "Day 3: Local culture"]
        }
    return data

def render_trace(trace: dict):
    st.subheader("🧭 Planning Trace (Concise)")
    with st.container(border=True):
        if trace.get("inputs_considered"):
            st.markdown("**Inputs Considered**")
            for it in trace["inputs_considered"][:5]:
                st.write(f"• {it}")
        if trace.get("planning_steps"):
            st.markdown("**Steps**")
            prog = st.progress(0)
            for i, step in enumerate(trace["planning_steps"][:5], start=1):
                st.write(f"{i}. {step}")
                prog.progress(int(i / len(trace["planning_steps"][:5]) * 100))
                time.sleep(0.1)
        if trace.get("final_structure"):
            st.markdown("**Itinerary Skeleton**")
            for i, s in enumerate(trace["final_structure"][:5], start=1):
                st.write(f"- {s}")


def stream_markdown(md_text: str):
    placeholder = st.empty()
    acc = ""
    for chunk in md_text.split():
        acc += chunk + " "
        placeholder.markdown(acc)
        time.sleep(0.015)


SUGGESTED_TRIPS = [
    {"destination": "Bali, Indonesia", "highlights": ["Ubud", "Uluwatu", "Nusa Penida"], "best_time": "Apr–Oct", "days": 5},
    {"destination": "Paris, France", "highlights": ["Louvre", "Montmartre", "Versailles"], "best_time": "Apr–Jun, Sep–Oct", "days": 4},
    {"destination": "Ladakh, India", "highlights": ["Pangong Tso", "Nubra Valley"], "best_time": "Jun–Sep", "days": 6},
    {"destination": "Kyoto, Japan", "highlights": ["Fushimi Inari", "Arashiyama"], "best_time": "Mar–May, Oct–Nov", "days": 3}
]

def render_suggested_trips():
    st.subheader("✨ Suggested Trips")
    for trip in SUGGESTED_TRIPS:
        with st.container(border=True):
            st.markdown(f"**{trip['destination']}** — Best: {trip['best_time']}, {trip['days']} days")
            st.markdown("Highlights: " + ", ".join(trip["highlights"]))

def main():
    st.title("🌍 AI Travel Guide – Personalized Tour Planner")
    st.caption("Built by Yashraj Limkar")

    # Sidebar
    st.sidebar.header("Trip Inputs")
    destination = st.sidebar.text_input("Destination", placeholder="e.g., Kyoto, Japan")
    date_from = st.sidebar.date_input("Start date", datetime.today())
    date_to = st.sidebar.date_input("End date", datetime.today() + timedelta(days=3))
    interests = st.sidebar.multiselect("Interests", ["Culture", "Food", "Nature", "Beaches", "Adventure"], default=["Culture"])
    budget = st.sidebar.selectbox("Budget", ["-", "Backpacker", "Mid-range", "Luxury"], index=2)
    openweather_key = st.sidebar.text_input("OpenWeather API Key", value=DEFAULT_OPENWEATHER_KEY, type="password")

    show_weather = st.sidebar.toggle("Show Weather", True)
    show_trace = st.sidebar.toggle("Show Planning Trace", True)
    stream_mode = st.sidebar.toggle("Stream Responses", True)

    tabs = st.tabs(["Itinerary Builder", "Suggested Trips", "Chat / Q&A"])

    # Itinerary Builder
    with tabs[0]:
        st.subheader("🧳 Itinerary Builder")
        generate_btn = st.button("Generate Itinerary", type="primary")

        if generate_btn:
            question = (
                f"Plan a day-wise itinerary for {destination} from {date_from} to {date_to}. "
                f"Include interests: {', '.join(interests)}. Budget: {budget}."
            )

            if show_trace:
                trace = generate_concise_trace(question)
                render_trace(trace)

            if show_weather and destination:
                render_weather_block(destination, (str(date_from), str(date_to)), openweather_key)

            if budget and destination:
                render_budget_section(destination, date_from, date_to, budget)

            st.subheader("🗺️ Generated Itinerary")
            try:
                qa_chain = get_qa_chain()
                response = qa_chain.invoke({"query": question})
                result = response["result"]
                answer_text = f"**Proposed Plan for {destination}**\n\n{result}"

                if stream_mode:
                    stream_markdown(answer_text)
                else:
                    st.markdown(answer_text)

            except Exception as e:
                st.error(f"Error: {e}")

    # Suggested Trips
    with tabs[1]:
        render_suggested_trips()

    # Chat / Q&A
    with tabs[2]:
        st.caption("Ask about destinations, travel tips, or itineraries.")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).markdown(msg["content"])

        prompt = st.chat_input("Ask your travel question...")
        if prompt:
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            if show_trace:
                trace = generate_concise_trace(prompt)
                render_trace(trace)

            try:
                qa_chain = get_qa_chain()
                response = qa_chain.invoke({"query": prompt})
                result = response["result"]

                if stream_mode:
                    with st.chat_message("assistant"):
                        stream_markdown(result)
                else:
                    st.chat_message("assistant").markdown(result)

                st.session_state.messages.append({"role": "assistant", "content": result})

            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
