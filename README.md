# Defense Hackathon Audio Sensing Network

Audio-based quadcopter sensing, in a mesh network.

A project for the USS Hornet Defense Tech Hackathon, 2024.

# Motivation

This is an attempt to build an acoustic tracking system for aerial vehicles. This is meant to be effective, low-cost, and able to pair with alternative sensor modalities (e.g., RF) for a better combined system.

Defense against drones (quadcopters or otherwise) first requires detecting them. This is typically done via their RF emissions, which can work over long distances---but when the drone is actively transmitting a detectable signal. RF tracking is not possible when a drone is sufficiently autonomous or has a sufficiently well-disguised RF signal. Camera-based tracking only works in good visibility conditions (not at night, for example), and radar has only limited effectiveness against smaller drones.

Audio is a promising alternative.

Any propeller-based drone emits audible noise (often ~70dB) from its propellers. Quiet propellers decrease this somewhat, but blade noise is impossible to mask completely.

So audio tracking should work on any drone.

The main downside is limited range: audio detection is only feasible out to perhaps 100m distances (i.e., a football field).

# Future Possibilities

- Drone fingerprinting via acoustic signature of their propellers
- Using time differences in sound reception at different sensors for localization via time of flight 
- Cheap sub-$100 solar-powered sensors with LoRa networking, capable of making a mesh with thousands of sensors

# Software Architecture

## Client

A Python-based script can run on any client with a microphone. Does on-device processing to detect whether a drone is present in real time.

## ML Model

We've built a small ML model capable of running on device.

## Server

Data is fed back to a central server for visualization.




