---
layout: post
title:  Computer Interfaced Non Invasive Photo-Plethysmographic Heart Rate Monitoring Device
date:   2017-04-20 00:10:45
categories: Self Initiated Project
excerpt_separator: <!--more-->
---
The objective of the project was to come up with a device and visualization software that can monitor the heart rate, temporal variation of the heart rate and the level of oxygen saturation in the blood in an non invasive manner. 
<!--more-->

The project was conducted by Ashwin de Silva and Sachini Hewage, initially for the Biomedical Engineering Section of EXMO'2017 Engineering and Technology Exhibition held at the University of Moratuwa. As hardware components of the device ATmega2560 chip and the Pulse Sensor Amped. For the software base, arduino IDE and Processing IDE was used. The coding was done in C++ with the use of arduino and processing libraries.


The voltage signal from the sensor it was sampled at 500 Hertz and this was done with help of the timer0 interrupt in ATmega2560. A  pulse threshold was defined as half of the difference between the signals' maximum and the minimum levels. The pulse threshold would updates after every pulse. When the signal level rises above the threshold, a pulse was detected and a delay was imposed from the point of pulse detection to prevent the method from detecting the dicrotic notch as a new pulse. A 10 - element queue was implemented to store the Inter-Beat-Interval (IBI) values such that at the end of every pulse, the new IBI would be enqueued and the earliest value would be dequeued. At every pulse, the method would calculate the average of the IBIs in the queue and would calculate the Pulse Rate using average IBI. 

Code    :   [[GitHub]](https://github.com/Laknath1996/PPG-Pulse-Meter) 

Poster  :   [[Poster]](https://drive.google.com/open?id=1XWz-zU3tcR34xWFqzaA9KgB3U79EDTWl)

<iframe width="480" height="270" src="https://www.youtube.com/embed/mOI8XzjkmrU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


