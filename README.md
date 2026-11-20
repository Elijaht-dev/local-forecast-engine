# Local Forecast Engine for Home Assistant OS

Add `https://github.com/Elijaht-dev/local-forecast-engine#ha-beta` under **Settings → Apps → App store → Repositories**. Install **Local Forecast Engine** and start it. The Supervisor builds the app locally from the sources in this branch. The first build downloads the Python dependencies and can take several minutes.

Install the [Local Forecast integration](https://github.com/Elijaht-dev/local-forecast/releases), restart Home Assistant and accept the discovered engine in **Settings → Devices & services**. Open **Forecast Lab** in the sidebar as an administrator.

This branch packages the [v0.1.0 engine source bundle](https://github.com/Elijaht-dev/local-forecast-engine/releases/tag/v0.1.0) for the app store. On Raspberry Pi 4, use a built-in baseline with this version. See the engine release notes for the PyTorch compatibility limitation.
