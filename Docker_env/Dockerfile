FROM python:3.8
COPY . /app
WORKDIR /app
EXPOSE 8501
RUN pip install libpython
RUN pip install -r requirements.txt
RUN pip install prophet
ENV MONGO_DB_URL="mongodb+srv://user:eq26pLfdsda@energyconsumption.fdvh4.mongodb.net/?retryWrites=true&w=majority"
CMD streamlit run pipeline_api.py --server.port $PORT