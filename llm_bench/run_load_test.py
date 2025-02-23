import os

model_name = "llama3p3-70b-instruct-fp8-engine"
user, input_tok, output_tok = 1, 6000, 350
token = "sk-xxxxxxx"
url = "http://localhost:8000/v1"

os.system(
    f"locust -t 2min -u {user} -r {user} -o {output_tok} -H {url} -p {input_tok} --api-key {token} --model={model_name} --prompt-randomize --chat --stream --provider openai --temperature 0.2 --summary summary.txt"
)
