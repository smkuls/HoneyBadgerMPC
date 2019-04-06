FROM dsluiuc/honeybadgermpc-base-image:latest

COPY . /usr/src/HoneyBadgerMPC

ARG BUILD
RUN pip install --no-cache-dir -e .[$BUILD]

RUN make -C apps/shuffle/cpp
