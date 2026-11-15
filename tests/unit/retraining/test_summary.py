from housing_mlops.retraining.summary import render_outcome as render


def test_render_with_gate():
    text = render(
        {
            "triggered": True,
            "reason": "forced retraining",
            "champion_version": "v1",
            "gate": {
                "candidate_version": "v2",
                "champion_version": "v1",
                "metric": "rmse",
                "candidate_value": 0.47,
                "champion_value": 0.49,
                "improvement": 0.0408,
                "promoted": True,
                "reason": "candidate rmse=0.4700 vs champion 0.4900 (+4.08%): promoted",
            },
        }
    )
    assert "## Retraining outcome" in text
    assert "| test | 0.4700 | 0.4900 | +4.08% | **True** |" in text
    assert "`v2`" in text


def test_render_without_gate_or_champion():
    text = render({"triggered": False, "reason": "no drift report", "champion_version": None})
    assert "Triggered: **False**" in text
    assert "`none`" in text
    assert "| test |" not in text


def test_render_without_champion_values():
    text = render(
        {
            "triggered": True,
            "reason": "no champion model; training the first one",
            "champion_version": None,
            "gate": {
                "candidate_version": "v1",
                "champion_version": None,
                "metric": "rmse",
                "candidate_value": 0.5,
                "champion_value": None,
                "improvement": None,
                "promoted": True,
                "reason": "no champion",
            },
        }
    )
    assert "| test | 0.5000 | n/a | n/a | **True** |" in text
