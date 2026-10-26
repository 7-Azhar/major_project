"""Both-page launcher opens the configured services without duplicate model startup."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

spec=importlib.util.spec_from_file_location('station_launcher',Path(__file__).resolve().parents[1]/'launch.py')
launcher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize('both,disabled,backend,count',[
    (True,False,'aura',2),(False,False,'aura',1),(True,True,'aura',0),(True,False,'native',1)])
def test_browser_selection_uses_configured_ports(monkeypatch,both,disabled,backend,count):
    browser=Mock();monkeypatch.setattr(launcher.webbrowser,'open',browser)
    launcher.open_workspaces(SimpleNamespace(port=8001,open_both=both,no_browser=disabled),
                             SimpleNamespace(video_backend=backend,aura_port=8767))
    assert browser.call_count==count
    if count: assert browser.call_args_list[0].args[0]=='http://127.0.0.1:8001'
    if count==2: assert browser.call_args_list[1].args[0]=='http://127.0.0.1:8767/'
