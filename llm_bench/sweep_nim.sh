# for u in 1 4 8 16 32 48; do
for u in 16 32; do
    locust -u $u -r 2 -p 5120 -o 275 -H http://0.0.0.0:8000 --provider nim --summary-file nim_summary_bf16.csv -t 2m --no-stream
done