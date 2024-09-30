import os

user, input_tok, output_tok = 10, 1000, 500
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1dWlkIjoiYzkxZWIzMjgtZjRkMS00MjhkLThlMWUtYjUwY2E1NTNkMTg5IiwiZXhwIjoxNzU5MTUxMDE1LCJvcmdfdXVpZCI6IjI4MDM2MDY4LWEzNjMtNDk5Ni1iMWJiLWIyYzJhZjk0ZTdiZiJ9.-h7EMVq51Y67DQ7zlNgeP9oubSDwqzig8zWrGQgi63g"
url = "https://http.llm-proxy.prod-models-default-us-west1.gcp.clusters.simplismart.ai"

# token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1dWlkIjoiZDUxODljNDAtNGQ4Ny00NjM4LWE5ODctY2JjZTY5Zjg4ZjY3Iiwib3JnX3V1aWQiOiI2ZDg0M2RiYy03NjUwLTQ2NjEtODQzNS0xYjI4MGI0MmI2NmUiLCJleHAiOjE3Mjc3MTE5NTN9.BPOuMmktkxSu2azDbM8CJLPNJxWkgPNdUQHQb9tsp6U"
# url = "https://http.llm-proxy.model-default-us-west-2.aws.staging.clusters.simplismart.ai"
os.system(
    f"locust -t 2min -u {user} -r {user} -o {output_tok} -H {url} -p {input_tok} --api-key {token} --model=llama3_1 --prompt-randomize --chat --stream --provider openai --temperature 0.2 --header id:f49b2e20-fef3-4441-9358-897f946b8ae2"
)