# Online-inference service container.
#
# Java analogy: this is the Dockerfile equivalent of packaging a Spring Boot app into a
# JRE base image — except instead of `COPY target/app.jar` you copy a Python source tree
# plus a joblib-serialized model file, and instead of `java -jar app.jar` the entrypoint
# starts an ASGI server (uvicorn) that hosts the FastAPI app.
#
# python:3.11-slim per research/NOTE-19-inference-serving.md's Docker guidance (use a
# 3.10+ base for compatibility with the pinned fastapi/uvicorn/pydantic versions).
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker's layer cache is reused across rebuilds that only
# change application code — the same reasoning as copying pom.xml/build.gradle before
# `mvn dependency:go-offline` in a Java Dockerfile, before copying the rest of the source.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the pre-trained model artifact. The model is trained and
# joblib-dumped as a separate build step (train_model.py, run before `docker build`),
# not inside the container at startup — the same reason you don't compile your Java
# source inside the runtime container on every boot.
COPY app.py model.joblib model_metadata.json ./

EXPOSE 8000

# --host 0.0.0.0 so the server accepts connections from outside the container's network
# namespace, the containerized equivalent of Tomcat binding to 0.0.0.0 instead of
# localhost. No --reload here (that's a local-dev-only flag per NOTE-19).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
