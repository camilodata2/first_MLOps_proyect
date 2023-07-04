FROM python:3.10.6-slim-buster

WORKDIR /app

COPY make_api/requirements.txt .

RUN pip install -U pip && pip install -r requirements.txt

COPY make_api/ ./make_api

#COPY model/model.pkl ./model/model.pkl

COPY initializer.sh .

RUN chmod +x initializer.sh

EXPOSE 8000

ENTRYPOINT ["./initializer.sh"]