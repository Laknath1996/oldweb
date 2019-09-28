---
layout: post
title:  Fully Analog Autonomous Line Following Robot
date:   2017-06-12 00:10:45
categories: Course Project
excerpt_separator: <!--more-->
---
Fully analog autonomous line follower robot was developed as a group project for the module EN2090. The objective was to implement the PID algorithm and Pulse Width Modulation using analog components. 
<!--more-->

A sensor panel comprised of 4 pairs of high intensity LEDs and LDRs was used. For each sensor input a specific weight was applied such that when the robot is moving along a straight white line in a black background the sum of all the weighted input signals is equal to zero. If a robot encounters a turn the summation would become non zero thus generating an error. This error was fed in to the PID circuit to generate the error signal. The error signal then is added/ subtracted from the motor base voltages depending on the correction requirement of the Robot in order to follow the line. These motor voltages are then passed down to the comparator together with the triangular wave coming from the triangular wave generator circuit to generate the PWM signal that would drive the motors. The PWM signal is passed through the transistors and fed in to the respective motors, thus making the correction to the robot’s trajectory.

Report  :   [[Report]](https://drive.google.com/open?id=1b5UxT9wwt_pV0F6SC4BJM60UuxMRrZad)

Grade   :   4.2/4.2

Supervisor(s)   :   Mr. Lahiru Chamain, Mr. Didula Dissanayaka 

<iframe width="480" height="270" src="https://www.youtube.com/embed/sUkAM-0J3dk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


