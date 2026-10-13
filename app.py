import joblib
import pandas as pd
import streamlit as st
from datetime import date
from pathlib import Path

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
)

st.title("🏨 Hotel Booking Cancellation Predictor")
st.write(
    "Estimate the probability of cancellation using the trained "
    "Random Forest pipeline from the Hotel Booking Demand project."
)

MODEL_PATH = Path(__file__).resolve().parent / "hotel_cancellation_pipeline.pkl"


# ---------------------------------------------------------
# Load the saved end-to-end sklearn pipeline
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Put `hotel_cancellation_pipeline.pkl` "
        "in the same folder as `app.py`."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

# The pipeline was trained with these exact input columns.
EXPECTED_FEATURES = list(model.feature_names_in_)

# Values learned by the fitted OneHotEncoder.
try:
    preprocessor = model.named_steps["preprocessor"]
    encoder = preprocessor.named_transformers_["cat"]
    categorical_transformer = next(
        item for item in preprocessor.transformers_ if item[0] == "cat"
    )
    categorical_columns = list(categorical_transformer[2])
    encoder_categories = {
        column: list(values)
        for column, values in zip(categorical_columns, encoder.categories_)
    }
except Exception as exc:
    st.error(f"Could not inspect the fitted preprocessing pipeline: {exc}")
    st.stop()


# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.subheader("Booking details")
st.caption(
    "Use information that would be available when the booking is made. "
    "The model returns an estimate, not a guarantee."
)

with st.form("booking_inputs"):
    st.markdown("#### 1. Hotel and arrival")

    col1, col2, col3 = st.columns(3)
    with col1:
        hotel_options = encoder_categories["hotel"]
        hotel = st.selectbox("Hotel type", hotel_options)
    with col2:
        arrival_date = st.date_input(
            "Expected arrival date",
            value=date(2026, 12, 15),
            min_value=date(2015, 1, 1),
            max_value=date(2035, 12, 31),
        )
    with col3:
        lead_time = st.number_input(
            "Days between booking and arrival",
            min_value=0,
            max_value=2000,
            value=30,
            step=1,
        )

    st.markdown("#### 2. Stay and guests")
    col1, col2, col3 = st.columns(3)
    with col1:
        weekend_nights = st.number_input(
            "Weekend nights", min_value=0, max_value=60, value=1, step=1
        )
    with col2:
        weekday_nights = st.number_input(
            "Weekday nights", min_value=0, max_value=60, value=2, step=1
        )
    with col3:
        adults = st.number_input(
            "Adults", min_value=0, max_value=20, value=2, step=1
        )

    col1, col2 = st.columns(2)
    with col1:
        children = st.number_input(
            "Children", min_value=0, max_value=20, value=0, step=1
        )
    with col2:
        babies = st.number_input(
            "Babies", min_value=0, max_value=10, value=0, step=1
        )

    st.markdown("#### 3. Booking channel and payment")
    col1, col2, col3 = st.columns(3)
    with col1:
        meal = st.selectbox("Meal plan", encoder_categories["meal"])
    with col2:
        market_segment = st.selectbox(
            "Market segment", encoder_categories["market_segment"]
        )
    with col3:
        distribution_channel = st.selectbox(
            "Distribution channel", encoder_categories["distribution_channel"]
        )

    col1, col2 = st.columns(2)
    with col1:
        deposit_type = st.selectbox(
            "Deposit type", encoder_categories["deposit_type"]
        )
    with col2:
        customer_type = st.selectbox(
            "Customer type", encoder_categories["customer_type"]
        )

    st.markdown("#### 4. Guest history and preferences")
    col1, col2, col3 = st.columns(3)
    with col1:
        is_repeated_guest = st.selectbox(
            "Repeat guest?", [0, 1], format_func=lambda value: "Yes" if value else "No"
        )
    with col2:
        previous_cancellations = st.number_input(
            "Previous cancellations", min_value=0, max_value=100, value=0, step=1
        )
    with col3:
        previous_bookings_not_canceled = st.number_input(
            "Previous bookings not canceled",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        reserved_room_type = st.selectbox(
            "Reserved room type", encoder_categories["reserved_room_type"]
        )
    with col2:
        adr = st.number_input(
            "Average daily rate (ADR)",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            step=10.0,
        )
    with col3:
        total_of_special_requests = st.number_input(
            "Special requests", min_value=0, max_value=20, value=0, step=1
        )

    col1, col2 = st.columns(2)
    with col1:
        required_car_parking_spaces = st.number_input(
            "Required car parking spaces",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
        )
    with col2:
        country_options = encoder_categories["country"]
        # Keep the complete set of categories learned during training.
        country_default = "PRT" if "PRT" in country_options else country_options[0]
        country = st.selectbox(
            "Guest country",
            country_options,
            index=country_options.index(country_default),
        )

    st.markdown("#### 5. Company and agent information")
    col1, col2 = st.columns(2)
    with col1:
        company_missing = st.selectbox(
            "Company information",
            [1, 0],
            format_func=lambda value: "Not available" if value == 1 else "Available",
        )
    with col2:
        agent_missing = st.selectbox(
            "Agent information",
            [1, 0],
            format_func=lambda value: "Not available" if value == 1 else "Available",
        )

    submitted = st.form_submit_button(
        "Predict cancellation", use_container_width=True
    )


# ---------------------------------------------------------
# Feature engineering and prediction
# ---------------------------------------------------------
if submitted:
    # These date conventions should match the training notebook.
    # Python weekday: Monday=0 ... Sunday=6.
    arrival_year = arrival_date.year
    arrival_month = arrival_date.strftime("%B")
    arrival_week_number = int(arrival_date.isocalendar().week)
    arrival_day_of_week = int(arrival_date.weekday())

    total_stay_nights = int(weekend_nights + weekday_nights)
    total_guests = int(adults + children + babies)
    zero_guests = int(total_guests == 0)
    total_guest_nights = int(total_guests * total_stay_nights)

    input_data = {
        "hotel": hotel,
        "lead_time": lead_time,
        "arrival_date_year": arrival_year,
        "arrival_date_month": arrival_month,
        "arrival_date_week_number": arrival_week_number,
        "stays_in_weekend_nights": weekend_nights,
        "stays_in_week_nights": weekday_nights,
        "adults": adults,
        "children": children,
        "babies": babies,
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": is_repeated_guest,
        "previous_cancellations": previous_cancellations,
        "previous_bookings_not_canceled": previous_bookings_not_canceled,
        "reserved_room_type": reserved_room_type,
        "deposit_type": deposit_type,
        "customer_type": customer_type,
        "adr": adr,
        "required_car_parking_spaces": required_car_parking_spaces,
        "total_of_special_requests": total_of_special_requests,
        "company_missing": company_missing,
        "agent_missing": agent_missing,
        "total_stay_nights": total_stay_nights,
        "total_guests": total_guests,
        "zero_guests": zero_guests,
        "total_guest_nights": total_guest_nights,
        "arrival_day_of_week": arrival_day_of_week,
    }

    input_df = pd.DataFrame([input_data])

    # Match the fitted pipeline's exact feature order.
    missing_columns = set(EXPECTED_FEATURES) - set(input_df.columns)
    extra_columns = set(input_df.columns) - set(EXPECTED_FEATURES)

    if missing_columns or extra_columns:
        st.error(
            f"Input schema mismatch. Missing: {sorted(missing_columns)}; "
            f"unexpected: {sorted(extra_columns)}"
        )
    else:
        input_df = input_df[EXPECTED_FEATURES]

        try:
            prediction = int(model.predict(input_df)[0])
            probabilities = model.predict_proba(input_df)[0]
            cancellation_probability = float(probabilities[1])
            no_cancellation_probability = float(probabilities[0])

            st.divider()
            st.subheader("Prediction result")

            col1, col2 = st.columns(2)
            col1.metric("Cancellation probability", f"{cancellation_probability:.1%}")
            col2.metric(
                "Non-cancellation probability", f"{no_cancellation_probability:.1%}"
            )

            if prediction == 1:
                st.warning("The model predicts that this booking will be canceled.")
            else:
                st.success(
                    "The model predicts that this booking will not be canceled."
                )

            st.progress(cancellation_probability)
            st.caption(
                "The class prediction uses the model's default decision threshold. "
                "Probabilities are estimates and should not be treated as guarantees."
            )

            with st.expander("Inspect the 30 model input features"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
