# Yu-Gi-Oh Card Price Tracker (ZAR)

Track and visualize Yu-Gi-Oh! card prices across vendors, converted into South African Rand (ZAR), using an automated pipeline and interactive dashboard.
Site link: https://ygo-price-tracker.streamlit.app/

---

##  Overview

This project:
- Fetches daily price (12:00pm SAST) data (EUR/USD) from the **YGOPRODeck v7 API**
- Converts to **ZAR** using ECB rates via the **Frankfurter API**
- Stores vendor and set-level history in a normalized **SQLite** database
- Saves the latest snapshot as a CSV (`prices_latest.csv`)
- Includes an interactive **Streamlit + Plotly** dashboard for dynamic exploration by card, vendor, date, and rarity

---

##  Getting Started

### Prerequisites

- Python 3.10+
- `venv` or virtual environment support

### Installation & Setup

```bash
git clone <repo-url>
cd ygo-price-tracker
python -m venv .venv
# Activate:
# macOS/Linux:
source .venv/bin/activate
# Windows:
.\.venv\Scripts\activate

pip install -r requirements.txt
```

## Contributions

Contributions welcome! To help out:

1. Fork the repo and create a feature branch
2. Make changes and test locally
3. Submit a pull request with your improvements


