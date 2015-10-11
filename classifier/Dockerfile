# Version: 0.1
FROM ipython/scipystack
MAINTAINER you "your email"
RUN easy_install pika
RUN easy_install pattern
RUN easy_install nltk
RUN easy_install gensim
COPY classifier.py /
CMD ["python", "/classifier.py","address of rabbitmq", "rabbit id", "rabbitpasswd", "address of scimlservice"]
EXPOSE 16666
