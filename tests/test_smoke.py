"""Smoke test for maintainiq."""
from pme.sources.synthetic import SyntheticFleetSource

def test_smoke():
    source = SyntheticFleetSource(seed=42, n_machines=2, hours=24)
    records = list(source.emit())
    assert len(records) > 0
    df = source.history()
    assert len(df) > 0
    print(f"maintainiq smoke test passed with {len(records)} events")

if __name__ == "__main__":
    test_smoke()
