FROM python:3.7-slim
WORKDIR /wl_pdf 
EXPOSE 8501
COPY . .
RUN apt-get update
RUN apt-get -y install curl
RUN apt-get install build-essential -y
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["pdf_app.py"]