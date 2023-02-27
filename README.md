## Code implementation for "DCA: Delayed Charging Attack on the Electric Shared Mobility System"
###  A Simulation Platform for Electric-Taxi Ride-Hailing System.
**Primary contributors: Shuocheng Guo (U Alabama) and Xinwu Qian (U Alabama)**

<img src="https://github.com/sguo28/multimodal-transit-viz/raw/main/dca_viz.gif" width="500" height="400">

(Red dots: EVCS under attack. Blue dots: EVCS removed for repair. Green dots: EVCS under normal operation.)

Key features: Routing, Charging, Repositioning, Matching, Cybersecurity module (attack and detection algorithm), interaction between EVs and EV charging stations (EVCSs).

### Preprint
We are happy to help if you have any questions. If you used any part of the code, please cite the following paper (see [Arxiv preprint](https://arxiv.org/abs/2302.01972))

@article{guo2023dca,
  title={DCA: Delayed Charging Attack on the Electric Shared Mobility System},
  author={Guo, Shuocheng and Chen, Hanlin and Rahman, Mizanur and Qian, Xinwu},
  journal={arXiv preprint arXiv:2302.01972},
  year={2023}
}

### Our previous work for Gasoline-Taxi-based Ride-Hailing System
This simulation platform is extended from our previous work "[DROP: Deep relocating option policy for optimal ride-hailing vehicle repositioning](https://www.sciencedirect.com/science/article/pii/S0968090X22003369)" that was accepted by Transportation Research Part C: Emerging Technologies (see [Github repo here](https://github.com/sguo28/DROP_Simulator)).

@article{qian2022drop, title={DROP: Deep relocating option policy for optimal ride-hailing vehicle repositioning}, author={Qian, Xinwu and Guo, Shuocheng and Aggarwal, Vaneet}, journal={Transportation Research Part C: Emerging Technologies}, volume={145}, pages={103923}, year={2022}, publisher={Elsevier} }

### Prerequisite

#### Data Sources
|Data|Link|
|-|-|
|EV charging station|[AFDC](https://afdc.energy.gov/fuels/electricity_locations.html#/find/nearest?fuel=ELEC)|
|OD demand|[NYCTLC](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)|

#### Data Preprocessing

The preprocessed large files can be fetched via [OneDrive](https://bama365-my.sharepoint.com/:f:/g/personal/sguo18_crimson_ua_edu/Etn1ZPhQ20ZNicLpYbCKyyQBy8QlqGfm4kmaEbIugTXiew?e=8uqPSw).

### How to run the code
#### 1. Clone the repo
```bash
git clone https://github.com/sguo28/DCA_Simulator.git
cd DCA_Simulator/code
```
#### 2. Download the data

Download the data from [OneDrive](https://bama365-my.sharepoint.com/:f:/g/personal/sguo18_crimson_ua_edu/Etn1ZPhQ20ZNicLpYbCKyyQBy8QlqGfm4kmaEbIugTXiew?e=8uqPSw) and put them in the `data` folder.

#### 3. Run the code
```bash
python main_cnn.py
```
