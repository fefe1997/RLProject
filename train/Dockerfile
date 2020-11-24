FROM python:3.8.6-slim as builder

COPY requirements.txt /tmp

RUN pip3 install --user -r /tmp/requirements.txt

FROM nvidia/cuda:10.2-base

COPY --from=builder /root/.local /root/.local

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /train

COPY utils.py .
COPY policy.py .
COPY model.py .
COPY train.py .


RUN apt-get update && apt-get install -y --no-install-recommends python3 libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/root/.local/bin:$PATH

CMD [ "python3", "train.py" ]
