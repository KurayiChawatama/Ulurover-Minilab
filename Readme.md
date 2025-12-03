# ULUROVER Minilab

<div style="text-align: center;">
    <img src="./Images/Readme/Ulurover Logo.svg" alt="ULUROVER Logo" title="ULUROVER LOGO" width="350"/>
</div>

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Components](#key-components)
  - [Weather Station](#1-weather-station)
  - [Onboard Minilab (Wetlab)](#2-onboard-minilab-wetlab)
  - [Rock Identification & Camera Module](#3-rock-identification--camera-module)
- [Team](#team)

## Overview

This repository contains the minilab project for the [ULUROVER Team](https://www.linkedin.com/company/ulurover-team/posts/?feedView=all) Science division at Uludağ University for the 2026 competition season.

### Project Goals

This project builds upon [last year's semi-autonomous minilab](https://github.com/KurayiChawatama/ChemNose-An-Automated-Gas-Detection-and-Statistical-Analysis-Platform), with a focus on achieving **complete autonomy** for Mars rover gas detection and analysis.

### Development Status

⚠️ **Work in Progress** - Expected completion: Early 2026

The documentation is actively being developed and may change frequently.

## System Architecture

The minilab consists of three interconnected sections that communicate with each other and share power:

- **Wetlab Box** - Controlled by Raspberry Pi 5
- **Weather Station** - Controlled by Arduino Nano
- **Drilling Mechanism** - Manual control via Jetson Orin (Rover's main computer)

## Key Components

### 1. Weather Station

The weather station provides real-time atmospheric monitoring from the highest point of the rover. Running on an Arduino Nano-based PCB, it continuously collects environmental data.

**Monitored Metrics:**

| Metric | Unit | Sensor |
|--------|------|--------|
| Temperature | °C | BME280 |
| Humidity | 100% Scale | BME280 |
| Pressure | Pa | BME280 |
| UV Index | 1-10 Scale | GUVA-S12SD |
| Methane Gas | ppm | MQ-4 |
| Hydrogen Gas | ppm | MQ-8 |
| Carbon Dioxide | ppm | MQ-135 |

**Features:**
- Data collection every 1-5 seconds
- Date/time tracking via DS3231 RTC Module
- CSV output format for easy data processing
- Two operating modes:
  - **Connected mode**: Direct connection to Raspberry Pi via USB, data processed using Python serial library
  - **Standalone mode**: Data written to SD card, powered by external Li-ion pack
- Data visualization capabilities: time-series graphs and geographical overlay on rover path

### 2. Onboard Minilab (Wetlab)

The wetlab is the core experimental platform, autonomously conducting soil and rock analysis. It comprises three integrated subsystems:

#### 2.1 Soil and Rock Weighing Mechanism

**Dual weighing system using 1KG sensors:**
- **Soil weight sensor**: Mounted under collection containers to monitor sample weight during successive additions, ensuring compliance with competition limits
- **Rock weight sensor**: External placement for initial rock verification before processing

**Workflow:**
1. Rock picked up by robotic arm and placed on external sensor
2. Weight verification against competition limits
3. If approved, arm places rock on trap door for barrel insertion
4. Rock falls into identification slot

#### 2.2 Solution-Based Soil Experiments

The most complex subsystem, performing two main experiment categories:

**Gas Evolution Test (Organic Matter Detection):**
- H₂O₂ + ketones (R-C=O) → CO₂ production
- MQ-135 CO₂ sensor monitors gas evolution
- CO₂ change proportional to organic matter content

**Metal Ion Testing (Up to 3 tests):**
- Solution-based color change reactions
- 2 metal ion tests + 1 pH test capability
- Competition-specific reagents (varies by ARC, ERC, URC requirements)

**Sample Collection Process:**
- Soil poured remotely from drill into main caching container
- Three funnels for three separate samples
- Each container split into major/minor sections
- Minor sections: equipped with MQ-135 sensors for gas testing
- Major sections: used for aqueous ion extraction

**Peristaltic Pump System (8 pumps total):**

| Pump # | Function |
|--------|----------|
| 1 | H₂O₂ tank → 3 minor cache compartments |
| 2 | Water tank → 3 major cache compartments |
| 3 | pH indicator tank → 3 test tubes |
| 4 | Metal ion 1 reagent tank → 3 test tubes |
| 5 | Metal ion 2 reagent tank → 3 test tubes |
| 6 | Cache major compartment 1 → 3 test tubes |
| 7 | Cache major compartment 2 → 3 test tubes |
| 8 | Cache major compartment 3 → 3 test tubes |

**Testing Procedure:**
1. Water pumped into major compartments to dissolve ions from soil
2. Waiting period during rover movement for ion dissolution
3. Solution extracted into test tubes (3 per sample)
4. Color change reagents added from external tanks (50mL Falcon tubes)
5. Reactions photographed by Raspberry Pi Camera V2
6. Images transmitted to scientists for analysis

#### 2.3 Rock Identification & Camera Module

**Rotating Drum System:**
- 10-compartment drum with transparent outer wall
- Holds test tubes and rocks
- 360° servo motor rotation for sequential imaging
- Pizza slice-shaped compartments

**Imaging System:**
- Raspberry Pi Camera V2 mounted on minilab interior wall
- Views one compartment at a time
- White LED for supplemental lighting
- Step-by-step photography of all compartments

**Analysis Capabilities:**
- Manual analysis by scientists at base station
- Automated rock identification algorithm
- Optional: HAILO 13 TOPS AI accelerator for autonomous image analysis on Pi

**Data Output:**
- Images transmitted to base for scientist review
- Real-time rock classification
- Test tube color change documentation 

## Team

**2026 Science Team:**

- [Kurayi Chawatama](https://github.com/KurayiChawatama) - Science Team Leader
- [Meryem Ergün](https://github.com/merimikris)
- [Toprak Alkaya](https://github.com/Topraka18k)