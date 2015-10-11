#!/home/dbgswarm/anaconda/bin/python
#this version of scimlservice is the "safe" version
# does a reconnect to the event queue every time
# it uses Pika to talk to rabbitmq and azure tables to store the data
# it requires three parameters.   see the dockerfile for details
# it waits on port 16666 so this port must be exposed externally if you want
#it to be visible externally

import socket
import pika
from bottle import route, request, run
from azure.storage import TableService, Entity, BlobService
import socket
import json
import time

class conobj:
        def __init__(self,pikaIP, rabbitid, rabbitpasswd, TableAccountName, TableKey ):
                self.ipad = socket.gethostname()
                self.pikaIP = pikaIP
                self.creds = pika.PlainCredentials(rabbitid, rabbitpasswd)
                self.connec = pika.BlockingConnection(pika.ConnectionParameters(pikaIP, 5672,'/',self.creds))
                self.chan = self.connec.channel()
                self.chan.queue_declare(queue='hello', auto_delete=False, exclusive=False)
                self.table_service = TableService(account_name=TableAccountName, account_key=TableKey)
                self.table_service.create_table('sciml')
                self.chan.basic_publish(exchange='',
                        routing_key='hello',
                        body='start up newest server at address '+self.ipad)
                self.chan.cancel()
                self.connec.close()
                self.chan = None

        def recon(self):
                #this is the slow method.   but reliable.
                try:
                        self.connec = pika.BlockingConnection(pika.ConnectionParameters(self.pikaIP, 5672,'/',self.creds))
                        self.chan = self.connec.channel()
                        self.chan.queue_declare(queue='hello', auto_delete=False, exclusive=False)
                except:
                        self.fullrecon()
        
        def fullrecon(self):
                if self.chan is not None:
                        self.chan.cancel()
                #self.connec.close()
                time.sleep(2.0)
                print "reconnecting to rabbitmq"
                self.connec = pika.BlockingConnection(pika.ConnectionParameters(self.pikaIP, 5672,'/',self.creds))
                self.chan = self.connec.channel()
                self.chan.queue_declare(queue='hello', auto_delete=False, exclusive=False)
                self.chan.basic_publish(exchange='', routing_key='hello',
                body='reconnecting to rabbitmq')
        
        def decon(self):
                if self.chan is not None:
                        #self.chan.basic_publish(exchange='',
                        #        routing_key='hello',
                        #        body='disconect from address '+self.ipad)
                        self.chan.cancel()
                        self.connec.close()
                        self.chan = None

mycon = None


@route('/get_answer')
def get_answer():
        if mycon.connec is None:
                return "no connection"
        else:
                return "connected"




@route('/echo_string', method='POST')
def echo_string():
        if mycon.chan is None:
                mycon.recon()
        st = request.forms.get('string')
        tim = str(time.time()%1000000)+" "
        cls = "none"
        #if this is a classification event it will contain "(scuebce topic)" near the start
        #of the string.  this is followed by the classification
        #the remainder of the string is ":" followed by the json image of the document
        #otherwise this is an initialization message
        xb = st.find("(")
        if xb < 0:
                # this is a wakeup call
                mycon.fullrecon()
                #mycon.chan.basic_publish(exchange='', routing_key='hello',
                #body=tim+'recieved='+st[0:60])  
        if xb >= 0 and xb < 3:
                x = st.find(')')
                topic = st[xb+1:x]
                st1 = st[x:]
                y = st1.find(" :")
                cls = st1[1:y]
                #print "cls ="+cls
                ddoc= st1[y+2:]
                #print "ddoc ="+ddoc
                dic = json.loads(ddoc)
                rk = str(hash(dic["title"]))
                item = {'PartitionKey': topic, 'RowKey': rk, 'class': cls, 
                'title': dic["title"], 'body': dic["doc"] }
                try:
                        mycon.table_service.insert_entity('sciml', item)
                except:
                        s= "table service error ... likely duplicate"
                        mycon.chan.basic_publish(exchange='', routing_key='hello', body=tim+s)
                
                try:
                        mycon.chan.basic_publish(exchange='', routing_key='hello',
                        body=tim+'the class ='+cls)
                except:
                        mycon.fullrecon()
                        mycon.chan.basic_publish(exchange='', routing_key='hello',
                        body=tim+'the class ='+cls)
        else:
                mycon.chan.basic_publish(exchange='', routing_key='hello',
                body=tim+'recieved='+st[0:60])      
        mycon.decon()
        return tim+"string was "+cls


#hn = socket.gethostname()
#run(host='0.0.0.0', port=16666)


import sys
import getopt

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
    
    if argv is None: 
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error, msg:
            raise Usage(msg)
       
        
        rabbitid = "fee"
        rabbitpasswd = "fi"
        pikaIP = 'fum'
        tableAccountName = 'boo'
        TableKey = "bar"
        if len(args)< 5:
            print "args needed are: IPaddress of Rabbitmq, rabbitid, rabbitpasswd, Azure Table account, azure key"
            return 2
        
        pikaIP = args[0]
        rabbitid = args[1]
        rabbitpasswd = args[2]
        TableAccountName = args[3]
        TableKey  = args[4]
        
        print "starting with RabbitMQ IP="+pikaIP+" rabbitid ="+rabbitid+" rabbitpasswd ="+rabbitpasswd+"table acount ="+TableAccountName
        print "and table key ="+TableKey
        global mycon
        mycon = conobj(pikaIP, rabbitid, rabbitpasswd, TableAccountName, TableKey)
        run(host='0.0.0.0', port=16666)

        print "all done."
        
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())
