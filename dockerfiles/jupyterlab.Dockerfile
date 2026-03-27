FROM quay.io/jupyter/scipy-notebook:latest

USER root

# Install Java 17 (required by Spark/PySparkling)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gnupg ca-certificates && \
    curl -fsSL https://packages.adoptium.net/artifactory/api/gpg/key/public \
      | gpg --dearmor -o /usr/share/keyrings/adoptium.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb bookworm main" \
      > /etc/apt/sources.list.d/adoptium.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends temurin-17-jre && \
    apt-get purge -y gnupg && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/lib/jvm/temurin-17-jre-* /usr/lib/jvm/temurin-17-jre

ENV JAVA_HOME=/usr/lib/jvm/temurin-17-jre

USER jovyan

# Pin JupyterLab and install ML/Spark dependencies
RUN pip install --no-cache-dir \
    jupyterlab==4.5.6 \
    "pyspark>=3.1,<3.2" \
    "h2o_pysparkling_3.1==3.46.0.6.1" \
    mlflow==3.10.1 \
    boto3 \
    psycopg2-binary \
    "setuptools<72"

# Download hadoop-aws JARs for S3 access (Spark 3.1 bundles Hadoop 3.2)
RUN PYSPARK_JARS=$(python -c "import pyspark; print(pyspark.__path__[0] + '/jars')") && \
    curl -fsSL -o "$PYSPARK_JARS/hadoop-aws-3.2.0.jar" \
      https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.2.0/hadoop-aws-3.2.0.jar && \
    curl -fsSL -o "$PYSPARK_JARS/aws-java-sdk-bundle-1.11.375.jar" \
      https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.375/aws-java-sdk-bundle-1.11.375.jar
