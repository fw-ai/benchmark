aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/o5q0o9k5

IMG_TAG=v7
docker buildx build --platform linux/amd64  -t adaptive/llm-bench:$IMG_TAG .
docker tag adaptive/llm-bench:$IMG_TAG public.ecr.aws/o5q0o9k5/adaptive/llm-bench:$IMG_TAG
docker push public.ecr.aws/o5q0o9k5/adaptive/llm-bench:$IMG_TAG