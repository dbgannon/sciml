# Version: 0.1.0
FROM ipython/scipystack
MAINTAINER your name "your email"
RUN easy_install azure
RUN easy_install pika
RUN easy_install bottle
COPY scimlservice.py /
CMD ["python", "/scimlservice.py", "ip of rabbitmq", "rabbit id", "rabbit passwd", "Azure storage account","storage account key"]
EXPOSE 16666
