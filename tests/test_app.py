"""Every dashboard page should load without errors."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app"
PAGES = [APP / "Home.py", *sorted((APP / "pages").glob("*.py"))]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_runs(page):
    at = AppTest.from_file(str(page), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


def test_customer_lookup_and_sliders():
    at = AppTest.from_file(str(APP / "pages" / "2_Customers.py"), default_timeout=60).run()
    at.text_input[0].input("CUST00001").run()
    assert not at.exception and len(at.metric) >= 7
    at = AppTest.from_file(str(APP / "pages" / "5_Inventory.py"), default_timeout=60).run()
    at.slider[0].set_value(0.99).run()
    assert not at.exception and at.metric[0].value == "2.33"
