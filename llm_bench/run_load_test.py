import os

user, input_tok, output_tok = 1, 100, 500
token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1dWlkIjoiYzkxZWIzMjgtZjRkMS00MjhkLThlMWUtYjUwY2E1NTNkMTg5IiwiZXhwIjoxNzU5MTUxMDE1LCJvcmdfdXVpZCI6IjI4MDM2MDY4LWEzNjMtNDk5Ni1iMWJiLWIyYzJhZjk0ZTdiZiJ9.-h7EMVq51Y67DQ7zlNgeP9oubSDwqzig8zWrGQgi63g"
url = "http://localhost:30000/v1"

os.system(
    f"locust -t 2min -u {user} -r {user} -o {output_tok} -H {url} -p {input_tok} --api-key {token} --model=llama3_1 --prompt-randomize --chat --stream --provider openai --temperature 0.2 --summary summary.txt"
)
