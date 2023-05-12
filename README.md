# Alleyway Car Safety System

Many neighborhoods are filled with narrow, dangerous alleyways where cars pool into. These alleyways have many blindspots, but are still a source of pedestrian and vehicle traffic.

This system aims to prevent accidents from occuring within these areas. A camera is placed at the end of the alleyway and looks upon any oncoming cars. This camera is connected to a RaspberryPi micro-computer. Cars, bikes, and pedestrians are identified in real-time from camera footage using the open source YOLO machine learning model.

Once an object (car, bike, etc.) is identified, the camera uses a novel front-facing speed algorithm to determine the speed of the vehicle. If a car's speed is exceeds 35 mph (the typical speed limit for suburban neighborhoods), its video is stored on a local hard drive for future use. Additionally, an external sensor (a light) will switch on when vehicles approach, signalling to any nearby pedestrians or vehicles of the oncoming object.

Here is a link to a video demonstrating the project: https://drive.google.com/file/d/1oGILD-yKC9IMxU1KhM-CrmRNbWFA-tdH/view?usp=sharing 
