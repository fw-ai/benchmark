import os

user, input_tok, output_tok = 10, 1000, 500

os.system(
    f"locust -t 2min -u {user} -r {user} -o {output_tok} -H http://localhost:8200 -p {input_tok} --api-key demo --model=llama3 --prompt-randomize --chat --provider openai"
)