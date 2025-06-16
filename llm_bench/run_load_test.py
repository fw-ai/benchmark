import os

model_name = "llama3p1"
users = [1,2,4,8]
input_toks = [64, 256, 512, 1024]
output_toks = [64, 256, 512, 1024]

for user in users:
    for input_tok in input_toks:
        for output_tok in output_toks:
            token = "sk-xxxxxxx"
            url = "http://localhost:8000/v1"

            os.system(
                f"locust -t 30sec -u {user} -r {user} -o {output_tok} -H {url} -p {input_tok} --api-key {token} --model={model_name} --prompt-randomize --chat --stream --provider openai --temperature 0.2 --summary summary-test.txt"
            )
