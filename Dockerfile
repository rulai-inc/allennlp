FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

WORKDIR /home/dialog/allennlp

EXPOSE 8080

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY *.py config.yml ./

CMD uvicorn api:app --host 0.0.0.0 --port 8080
