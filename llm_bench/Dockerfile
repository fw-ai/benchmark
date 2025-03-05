FROM python:3.10-bookworm
RUN apt-get update -qqy  && apt-get install curl jq -qqy

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /

COPY . .
RUN chmod +x entrypoint.sh
RUN chmod +x launch_all.sh

ENTRYPOINT ["/entrypoint.sh"]