---
layout: post
title:  Custom Processor implemented on FPGA for Image Downsampling
date:   2018-05-25 00:10:45
categories: Course Project
excerpt_separator: <!--more-->
---
Field Programmable Gate Array is a versatile platform for digital design. In this course project, a custom built processor was developed on FPGA in order to downsample an image. 
<!--more-->

The algorithm was first designed and simulated on MATLAB to verify its credibility.Then the processor requirements such as the number of registers, buses, data memory size, instruction memory size, word sizes etc was determined alongside with the Instruction Set Architecture (ISA) and the Assembly codes. After the processor specifications were fixed the coding was completed using Verilog HDL which was followed by debugging and testing on hardware. 

An RS232 cable (UART based) was used to establish the communication between the PC and the FPGA. Python scripts were used to mediate the data transmitting and data receiving. 

 <iframe src="https://youtu.be/ZEHNLUbt3R4"
   width="560" height="315" frameborder="0" allowfullscreen></iframe>


