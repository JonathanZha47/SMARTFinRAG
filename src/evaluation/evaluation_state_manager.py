import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Define the key used to store the DataFrame in session state
RESPONSE_QUALITY_STATE_KEY = "response_quality_df"

def initialize_response_quality_state():
    """Initializes the response quality DataFrame in session state if it doesn't exist."""
    if RESPONSE_QUALITY_STATE_KEY not in st.session_state:
        st.session_state[RESPONSE_QUALITY_STATE_KEY] = pd.DataFrame()
        logger.info(f"Initialized empty DataFrame with key '{RESPONSE_QUALITY_STATE_KEY}' in session state.")

def get_response_quality_df() -> pd.DataFrame:
    """Retrieves the cumulative response quality DataFrame from session state."""
    initialize_response_quality_state() # Ensure it's initialized before access
    return st.session_state[RESPONSE_QUALITY_STATE_KEY]

def append_response_quality_result(new_result_df: Optional[pd.DataFrame]):
    """Appends a new result DataFrame to the cumulative DataFrame in session state."""
    initialize_response_quality_state() # Ensure state exists
    if new_result_df is not None and not new_result_df.empty:
        try:
            current_df = st.session_state[RESPONSE_QUALITY_STATE_KEY]
            updated_df = pd.concat([current_df, new_result_df], ignore_index=True)
            st.session_state[RESPONSE_QUALITY_STATE_KEY] = updated_df
            logger.info(f"Appended {len(new_result_df)} row(s) to '{RESPONSE_QUALITY_STATE_KEY}'. New total: {len(updated_df)} rows.")
        except Exception as e:
            logger.error(f"Failed to append results to DataFrame in session state: {e}", exc_info=True)
    elif new_result_df is None:
        logger.warning("Attempted to append None DataFrame to response quality history.")
    else: # new_result_df is empty
        logger.warning("Attempted to append empty DataFrame to response quality history.")


def clear_response_quality_history():
    """Clears the response quality DataFrame stored in session state."""
    if RESPONSE_QUALITY_STATE_KEY in st.session_state:
        st.session_state[RESPONSE_QUALITY_STATE_KEY] = pd.DataFrame()
        logger.info(f"Cleared response quality history DataFrame ('{RESPONSE_QUALITY_STATE_KEY}') in session state.")
    else:
        # If not present, initialize it as empty anyway
        initialize_response_quality_state()
        logger.info("Response quality history key not found, initialized empty DataFrame.")


def calculate_cumulative_response_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates aggregate metrics from the cumulative response quality DataFrame."""
    if df is None or df.empty:
        # Return default structure when empty
        return {
            'total_evaluations': 0,
            'avg_faithfulness': np.nan,
            'avg_relevancy': np.nan,
            'faithfulness_pass_rate': np.nan,
            'relevancy_pass_rate': np.nan,
            'evaluation_errors': 0,
        }

    # Ensure columns exist before calculating stats
    metrics = {
        'total_evaluations': len(df),
        'avg_faithfulness': df['faithfulness'].mean(skipna=True) if 'faithfulness' in df else np.nan,
        'avg_relevancy': df['relevancy'].mean(skipna=True) if 'relevancy' in df else np.nan,
        'faithfulness_pass_rate': (df['is_faithful'] == True).mean(skipna=True) if 'is_faithful' in df else np.nan,
        'relevancy_pass_rate': (df['is_relevant'] == True).mean(skipna=True) if 'is_relevant' in df else np.nan,
        'evaluation_errors': df['error'].notna().sum() if 'error' in df else 0 # Count rows where error is not None/NaN
    }
    return metrics 