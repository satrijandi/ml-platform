FROM python:3.11-slim

# Install Java 11 (required by H2O)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gnupg ca-certificates && \
    curl -fsSL https://packages.adoptium.net/artifactory/api/gpg/key/public \
      | gpg --dearmor -o /usr/share/keyrings/adoptium.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb bookworm main" \
      > /etc/apt/sources.list.d/adoptium.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends temurin-11-jre && \
    apt-get purge -y gnupg && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/lib/jvm/temurin-11-jre-* /usr/lib/jvm/temurin-11-jre

ENV JAVA_HOME=/usr/lib/jvm/temurin-11-jre

RUN pip install --no-cache-dir \
    h2o==3.46.0.6 \
    mlflow==3.10.1 \
    flask \
    boto3 \
    "setuptools<72"

WORKDIR /opt/serve
EXPOSE 8000

ENTRYPOINT ["python"]
CMD ["/opt/scripts/serve.py"]
