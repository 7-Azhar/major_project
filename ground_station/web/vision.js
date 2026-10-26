'use strict';
(() => {
  const element = id => document.getElementById(id);
  let snapshot = null, busy = false;
  // Use the configured service port, including installations with custom ports.
  async function loadVisionAddress() {
    try {
      const response = await fetch('/api/system', {signal: AbortSignal.timeout(10000)});
      if (!response.ok) throw Error('Station settings unavailable');
      const system = await response.json();
      if (!Number.isInteger(system.aura_port) || system.aura_port < 1024 || system.aura_port > 65535) throw Error('Invalid vision port');
      const url = `http://127.0.0.1:${system.aura_port}/`;
      document.querySelectorAll('[data-vision-external]').forEach(link => {
        link.href = url;
        link.removeAttribute('aria-disabled');
      });
      element('vision-url').textContent = url;
    } catch (error) {
      element('vision-url').textContent = 'Address unavailable — refresh to retry';
    }
  }
  void loadVisionAddress();
  window.addEventListener('station-connected', loadVisionAddress);
  window.addEventListener('station-disconnected', () => {
    element('vision-power').disabled = true;
    element('vision-summary').textContent = 'Ground station disconnected. Reconnecting…';
  });
  element('vision-open').onclick = () => {
    const frame = element('vision-frame');
    if (!frame.getAttribute('src')) frame.src = '/vision/';
    element('vision-dialog').showModal();
  };
  element('vision-close').onclick = () => element('vision-dialog').close();
  // Closing the dialog also releases its long-poll connection. The shared
  // pipeline keeps running; reopening loads its current settings again.
  element('vision-dialog').addEventListener('close', () => element('vision-frame').removeAttribute('src'));
  element('vision-power').onclick = async () => {
    if (!snapshot || busy) return;
    busy = true;
    element('vision-power').disabled = true;
    try {
      const session = await (await fetch('/api/session', {signal: AbortSignal.timeout(5000)})).json();
      const response = await fetch('/vision/api/control', {method: 'POST',
        signal: AbortSignal.timeout(10000),
        headers: {'Content-Type': 'application/json', 'X-Station-Token': session.token},
        body: JSON.stringify({powered: !snapshot.config.powered})});
      if (!response.ok) throw Error('Could not change vision power. Open Vision workspace to reconnect.');
    } catch (error) { element('vision-summary').textContent = error.message; }
    finally { busy = false; }
  };
  window.addEventListener('station-state', event => {
    const state = event.detail, video = state.video;
    const available = video.backend === 'aura' && video.connected;
    snapshot = video.aura;
    element('vision-power').disabled = busy || !available;
    element('vision-open').disabled = video.backend !== 'aura';
    element('vision-power').textContent = snapshot?.config.powered ? 'Vision power off' : 'Vision power on';
    element('vision-summary').textContent = video.backend !== 'aura' ? 'Native demo video. Launch the integrated configuration to use AURA.' :
      `${video.status} · ${video.source || 'No input'} · ${snapshot?.config.enhancement || 'raw'} · ${snapshot?.config.detection || 'off'} · ${snapshot?.excluded_count || 0} excluded`;
    // The main renderer owns telemetry/safety status; do not replace its warnings.
    if (element('sim-person')) element('sim-person').disabled = video.backend === 'aura';
  });
})();
