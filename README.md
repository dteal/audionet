# Defense Hackathon Audio Sensing Network

Sound-based quadcopter detection, in a mesh network.

A project for the USS Hornet Defense Tech Hackathon, 2024.

# Motivation

This is an attempt to build an acoustic tracking system for aerial vehicles. This is meant to be effective, low-cost, and able to pair with alternative sensor modalities (e.g., RF) for a better combined system.

Defense against drones (quadcopters or otherwise) first requires detecting them. This is typically done via their RF emissions, which can work over long distances---but only when the drone is actively transmitting a detectable signal. RF tracking is not possible when a drone is sufficiently autonomous or has a sufficiently well-disguised RF signal. Camera-based tracking only works in good visibility conditions (not at night, for example), and radar has only limited effectiveness against smaller drones.

Audio is a promising alternative.

Any propeller-based drone emits audible noise (often ~70dB) from its propellers. Quiet propellers decrease this somewhat, but blade noise is impossible to mask completely.

So audio tracking should work on any drone.

The main downside is limited range: audio detection is only feasible out to perhaps 100m distances (i.e., a football field)---sound decreases with 1/distance (6dB every doubled distance). To solve this, we propose using a network of cheap sensors.

# Future Possibilities

- Drone fingerprinting via acoustic signature of their propellers
- Using time differences in sound reception at different sensors for localization via time of flight 
- Cheap sub-$100 solar-powered sensors with LoRa networking, capable of making a mesh with thousands of sensors

# Software Architecture

## Client / Sensor Node

A Python-based script can run on any client with a microphone. Does on-device processing to detect whether a drone is present in real time.

## ML Model

We've built a small ML model capable of running on device.

## Server

Data is fed back to a central server for visualization and logging.

# Math

What are the theoretical limits of this approach?

## Sensor Coverage

Drone sound varies very significantly with propellers, wind levels, etc. We can estimate a single mic might have up to a ~200m=600ft range assuming a drone generates ~70dB sound pressure at 2m distance and the system can reliably detect sounds (limited by the noise floor) down to 30dB (sound pressure ~1/r, so decreases 6dB every doubled distance). Internet anecdotes confirm a ballpark ~100m range (https://forum.dji.com/thread-285772-1-1.html https://nextech.online/drone-noise-levels/). (In the traditional spirit of weird American distance units, about one sensor every football field = 91m). For example, sparsely covering an area the size of the main Berkeley university campus (600 x 1000m) at a 100m pitch probably requires at least 6x10=60 sensors. Round up to 100 sensors. At $100/sensor, this is only $10k.

## Sensor Construction

The following should be feasible to build a single sensor node:

Total cost: $100-ish. Microphone + microcontroller + battery + solar + radio, or smartphone + solar.

Battery life: days, or infinite with solar.

If optimized, power draw can be as low as ~150mW total. This power draw can come from (a) microphones and processing or (b) the radio. Microphones take about zero power, but real-time audio processing is heavy. Fortunately, simple audio processing can work on something as small as an ESP32-S3 microcontroller (real-time voice processing: <https://www.espressif.com/en/solutions/audio-solutions/esp-skainet/overview>), which draws ~100mW power when active. For the radio: LoRa radio is specifically designed to be long-range and power-efficient enough to last ~10 years on a coin cell battery (assuming intermittent transmission), so it might actually take less power than the processing. Give it another ~50mW average.

A battery would need ~13kJ=3.6Wh of storage to run the sensor per day. A single AA battery holds about ~1.5Wh of power. Smartphone batteries hold 5-10Wh, enough to run for 1.5-3 days. Stack multiple batteries for more time.

We can get more power with solar. A standard solar cell can provide ~50W/m^2 average per day, so a 60x60mm solar cell is enough to power the sensor continuously (assuming enough battery storage to last overnight).

