import random

from paho.mqtt import client as mqtt_client


class Mqtt_Node():
    def __init__(self,port,broker):
        self.port=port
        self.broker=broker
        self.client_id=f'publish-{random.randint(0, 1000)}'
        self.coordinate_topic="tulip/coordinate"
        self.error_topic="tulip/error"
        # Broker Connection 
        self.ack_topic = "tulip/ack"
        self.running_topic = "tulip/running"
        self.confarmation_topic = "tulip/confarmation"

    def connect_mqtt(self):
        def on_connect(client, userdata, flags, rc,properties):
            if rc == 0:
                print("Connected to MQTT Broker!")
            else:
                print("Failed to connect, return code %d\n", rc)
        self.client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2, self.client_id)
        # client.username_pw_set(username, password)
        self.client.on_connect = on_connect
        self.client.connect(self.broker, self.port)
        
    def publish(self,value,topic):
        if(topic==self.coordinate_topic):
            msg=f"coordinates:{value}"
            #msg=f"coordinates:{value[0]},{value[1]},{value[2]}"
        if(topic==self.error_topic):
            msg=f"Error: {value}"
        if(topic == self.ack_topic):
            # msg = f"Ack:{value}"
            msg = value
        if(topic == self.running_topic):
            # value - stop
            msg = f"Running:{value}"


        result = self.client.publish(topic, msg)
        # result: [0, 1]
        status = result[0]
        if status == 0:
            print(f"Send `{msg}` to topic `{topic}`")
        else:
            print(f"Failed to send message to topic {topic}")
        

    def start_listining(self):
        self.client.loop_start()
    
    def stop_listining(self):
        self.client.loop_stop()


    def subscribe(self,app,topic):
        global data
        def on_message(client, userdata, msg):
            print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
            app.show_box(msg.payload.decode())
            # app.show_box("Connection Succesfully")
            # data.append(msg.payload.decode())

        self.client.subscribe(topic)
        self.client.on_message = on_message
