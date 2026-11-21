# Local Forecast Engine

Start this app, install the **Local Forecast** integration from its custom Marketplace repository, and accept the discovered integration in **Settings → Devices & services**. Open **Forecast Lab** from the sidebar. Select a recorded entity and install a model in the Lab, or start with a baseline.

To show the forecast on a dashboard, select **Publish forecast** in the saved Lab project and add its sensor to the optional **Local Forecast Card**. [Card installation and visual editor](https://github.com/Elijaht-dev/local-forecast-card#install-the-public-beta)

The app automatically generates its pairing token in its private data directory. Set the optional API token to a generated value with at least 24 ASCII characters if you need manual pairing. The default networking configuration makes the API available on the Supervisor network.

**Retention days** controls the age of cached observations, input snapshots, and completed runs. **Threads** controls PyTorch CPU threads. Start with Timestral-5M and two threads on a Raspberry Pi.

App backups contain the SQLite database and model cache in the app data directory. Stop the app before manually copying the data directory so the database and WAL are consistent.
