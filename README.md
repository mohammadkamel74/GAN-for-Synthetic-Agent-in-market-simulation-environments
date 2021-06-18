# GAN-for-Synthetic-Agent-in-market-simulation-environments

## Introduction

In our project we have created a financial agent that buys and sells stocks on a simulated market by using Generative Adversarial Network. We have worked on Amazon, Apple, Goggle, Intel and Microsoft level 1 LOBSTER’s data. We consider each data set as on agent, in this case we would have 5 different agents which each line in LOBSTER’s data is one data sample regarding the agent. For example Amazon would be an agent and each line of the order book would be a data sample. Our final goal is to make a new agent based one the previous agents who sell and buy stocks in the market. To test the functionality of the generated agents we used Agent-Based Interactive Discrete Event Simulation environment([ABIDES](https://github.com/abides-sim/abides)).

## Data

We implemented our model by using the [LOBSTER](https://lobsterdata.com/info/DataSamples.php) data-sets which is based on the official NASDAQ Historical, named AMZN,AAPL, GOOG, INTC, MSFT and in this project we have used level one of each dataset. 

## Preprocessing

 * Filtered transactions with type equals to 4 and 5
 * dropped useless columns such as ’orderid’, ’eventtype’ and etc
 * Merging all 5 data-sets
 * shuffled the data
 * Min-max normalization or denormalization
 
After performing Preprocessing step we would have 123985 data with 3 features (Price, Direction and Size)

## Generative Adversarial Network

## Abides

ABIDES is an Agent-Based Interactive Discrete Event Simulation environment. ABIDES is designed from the ground up to support AI agent research in market applications. While simulations are certainly available within trading firms for their own internal use, there are no broadly available high-fidelity market simulation environments. ABIDES is designed from the ground up to support AI agent research in market applications. ABIDES enables the simulation of many trading agents interacting with an exchange agent to facilitate transactions.
