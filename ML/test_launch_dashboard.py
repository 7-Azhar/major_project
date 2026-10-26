"""Vision-only launch must reuse a running model and respect browser selection."""
import sys
from unittest.mock import Mock
import pytest
import launch_dashboard as launcher


@pytest.mark.parametrize('no_browser',[False,True])
def test_existing_vision_reused_without_loading_or_binding(monkeypatch,no_browser):
    monkeypatch.setattr(sys,'argv',['launch_dashboard.py']+(['--no-browser'] if no_browser else []))
    monkeypatch.setattr(launcher,'health',lambda port:{'application':'AURA'})
    browser=Mock();monkeypatch.setattr(launcher.webbrowser,'open',browser)
    socket=Mock(side_effect=AssertionError('Must reuse existing service'))
    monkeypatch.setattr(launcher.socket,'socket',socket)
    launcher.main()
    assert browser.call_count == (0 if no_browser else 1)
    if not no_browser: browser.assert_called_once_with('http://127.0.0.1:8766/')


def test_unrelated_occupied_port_is_not_reused(monkeypatch):
    monkeypatch.setattr(sys,'argv',['launch_dashboard.py','--no-browser'])
    monkeypatch.setattr(launcher,'health',lambda port:{'application':'Other'})
    monkeypatch.setattr(launcher.socket,'socket',Mock(side_effect=OSError('port occupied')))
    with pytest.raises(OSError,match='occupied'):launcher.main()
