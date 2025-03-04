FROM python:3.10-bookworm
COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /

COPY . .
RUN chmod +x entrypoint.sh
RUN chmod +x launch_all.sh

ENTRYPOINT ["/entrypoint.sh"]