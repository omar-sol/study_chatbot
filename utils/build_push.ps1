# Define the project ID variable
$ProjectId = "socialgpt-380714"

# Build the Docker image for the amd64 platform
docker build --platform linux/amd64 -t school_chunks .

# Tag the Docker image
docker tag school_chunks gcr.io/socialgpt-380714/school_chunks:latest

# Push the Docker image to Google Container Registry
docker push gcr.io/socialgpt-380714/school_chunks:latest