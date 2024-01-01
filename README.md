# Detection of APS Failure at Scania Trucks with Machine Learning.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack & Infrastructure](#tech-stack--infrastructure)
- [Project Architecture](#project-architecture)
- [Deployment Architecture](#deployment-architecture)
- [Manual Steps to Run the Application](#manual-steps-to-run-the-application)
- [Run the Application on the Cloud - AWS Deployment Steps](#run-the-application-on-the-cloud---aws-deployment-steps)

## Overview

The Air Pressure System (APS) is a critical component of a heavy-duty vehicle that uses compressed air to force a piston that provides pressure to the brake pads, thus slowing the vehicle down. The benefits of using an APS instead of a hydraulic system are easy availability and long-term sustainability of natural air. It is a Binary Classification problem where the dataset's positive class consists of component failures for a specific component of the APS system. The negative class consists of trucks with failures for components not related to the APS.

## Dataset

APS Failure at Scania Trucks. (2017). UCI Machine Learning Repository. [https://doi.org/10.24432/C51S51](https://doi.org/10.24432/C51S51).

## Tech Stack & Infrastructure

1. Python
2. Machine Learning
3. FastAPI
4. MongoDB
5. Docker
6. Apache Airflow
7. GitHub Actions
8. AWS EC2, ECR & S3

## Project Architecture

![image](flowcharts/project-architecture.png)

## Deployment Architecture

![image](flowcharts/deployment-architecture.png)

## Manual Steps to Run the Application

#### Step 1: Create a Virtual Environment and Install Dependency.

```bash
# Clone the Repository.
git clone https://github.com/gangulyaritra/aps-failure-detection.git

# Create a Virtual Environment.
python3 -m venv venv

# Activate the Virtual Environment.
source venv/bin/activate

# Install the Dependencies.
pip install -r requirements.txt
```

#### Step 2: Install the latest version of the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

#### Step 3: Export the Environment Variables.

```bash
=========================================================================
Paste the following credentials as environment variables.
=========================================================================

export AWS_ACCESS_KEY_ID="<AWS_ACCESS_KEY_ID>"
export AWS_SECRET_ACCESS_KEY="<AWS_SECRET_ACCESS_KEY>"
```

#### Step 4: Setup MongoDB and export dataset into MongoDB Collections.

```bash
python get_data.py
```

#### Step 5: Run the Application Server.

```bash
python main.py
```

#### Step 6: Containerize the Application using Docker.

```bash
docker build -t aps:latest --build-arg AWS_ACCESS_KEY_ID="<AWS_ACCESS_KEY_ID>" --build-arg AWS_SECRET_ACCESS_KEY="<AWS_SECRET_ACCESS_KEY>" --no-cache .
```

#### Step 7: Run the Docker Image in the local system.

```bash
docker run -p 8000:8000 aps
```

#### Step 8: Schedule Machine Learning Pipeline with Apache Airflow.

Steps to Run Airflow in Docker - [**Docs**](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html).

```bash
# Initialize the Airflow Database.
docker-compose up airflow-init

# Clean up the environment.
docker-compose down --volumes --remove-orphans

# Running Airflow inside Docker.
docker-compose up
```

#### Step 9: Launch the Airflow UI.

```bash
http://localhost:8080
```

- The default account has the login **airflow** and the password **airflow**.

- Set Airflow Environment Variable: `AIRFLOW__CORE__ENABLE_XCOM_PICKLING = True`

## Run the Application on the Cloud - AWS Deployment Steps

1. Log in to the AWS console.
2. Create an IAM user for deployment.

   - **With specific access:**
     1. **Amazon Elastic Compute Cloud (Amazon EC2):** AWS EC2 provides scalable computing capacity in the AWS Cloud and eliminates the need to invest in hardware. Hence, we can develop and deploy applications faster. EC2 can launch as many or as few virtual servers.
     2. **Amazon Elastic Container Registry (Amazon ECR):** AWS ECR is a container image registry service that is secure, scalable, and reliable. ECR supports private repositories with resource-based permissions. We use CLI to push, pull, and manage Docker images inside ECR.
     3. **Amazon Simple Storage Service (Amazon S3):** AWS S3 is an object storage service offering industry-leading scalability, data availability, security, and performance. We can store, organize, and protect data for virtually any use case, such as data lakes, cloud-native applications, and mobile apps.
   - **Description: About the deployment.**
     1. Build a Docker image of the source code.
     2. Push the Docker image to AWS ECR.
     3. Launch the AWS EC2 instance.
     4. Pull the Docker image from ECR to EC2.
     5. Launch the Docker image inside AWS EC2.
   - **Policy:**
     1. AmazonEC2ContainerRegistryFullAccess
     2. AmazonEC2FullAccess
     3. AmazonS3FullAccess

3. Create the access keys for IAM users.
4. Setup the S3 Bucket to store project artifacts and log files.
5. Create an ECR repository to store/save the docker image.
6. Launch an EC2 virtual machine with Ubuntu OS.
7. Connect the EC2 and install Docker inside the EC2 machine.

   ```bash
   sudo apt-get update && sudo apt-get upgrade -y

   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   newgrp docker
   ```

8. Configure EC2 as a self-hosted runner.

   `setting > actions > runner > new self hosted runner > choose OS > then run command one by one`

9. Setup GitHub Actions Secrets.

   ```bash
   =========================================================================
   Paste the following credentials as GitHub Actions Secrets.
   =========================================================================

   AWS_ACCESS_KEY_ID=AKIAU6GNCIA3P4UP5M6S
   AWS_SECRET_ACCESS_KEY=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
   AWS_REGION=ap-south-1
   AWS_ECR_LOGIN_URI=339732480054.dkr.ecr.ap-south-1.amazonaws.com
   ECR_REPOSITORY_NAME=aps
   ```

10. Create a new EC2 Security Group (Inbound rules) to access port **8000**.

## Authors

- [Aritra Ganguly](https://in.linkedin.com/in/gangulyaritra)

## License & Copyright

[MIT License](LICENSE)
