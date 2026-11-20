# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY constraints.txt ./
RUN pip install --no-cache-dir --constraint constraints.txt \
    --index-url https://download.pytorch.org/whl/cpu "torch==2.10.0+cpu"
COPY pyproject.toml LICENSE ./
COPY local_forecast_engine/ ./local_forecast_engine/
RUN pip install --no-cache-dir --constraint constraints.txt ".[models]" && pip check

FROM python:3.12-slim-bookworm
LABEL io.hass.type="addon" org.opencontainers.image.title="Local Forecast Engine" \
    org.opencontainers.image.source="https://github.com/Elijaht-dev/local-forecast-engine" \
    org.opencontainers.image.licenses="MIT"
ENV PATH="/opt/venv/bin:$PATH" LF_HOST="0.0.0.0" LF_DATA_DIR="/data" LF_PORT="8099" \
    PYTHONUNBUFFERED="1"
COPY --from=build /opt/venv /opt/venv
COPY LICENSE /usr/share/doc/local-forecast/LICENSE
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8099
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8099/healthz', timeout=2).close()"
ENTRYPOINT ["python", "-m", "local_forecast_engine"]
