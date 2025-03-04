aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/o5q0o9k5
docker buildx build --platform linux/amd64 -t adaptive-gitops/cli .
docker tag adaptive-gitops/cli:latest public.ecr.aws/o5q0o9k5/adaptive-gitops/cli:latest
docker push public.ecr.aws/o5q0o9k5/adaptive-gitops/cli:latest