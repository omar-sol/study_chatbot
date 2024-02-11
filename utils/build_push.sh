docker build --platform linux/amd64 -t school_chunks .
docker tag school_chunks gcr.io/PROJECT_ID/school_chunks:latest
docker push gcr.io/PROJECT_ID/school_chunks:latest