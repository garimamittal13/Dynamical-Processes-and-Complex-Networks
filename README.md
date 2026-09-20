# DPCN Assignment 1: Opinion Network Formation

**Course:** Dynamical Processes in Complex Networks (DPCN)

**Team Name:** Mai Vapas Aaunga

This repository contains the Python implementation, dataset, and final LaTeX report for our structural analysis of class opinions. The project constructs a complex network from survey data to uncover the latent ideological relationships across Technology, Education, Ethics, and Environment.

## Repository Contents

* `code.py`: The main Python script that cleans the data, calculates the Pearson correlation matrix, generates the network graph, computes centrality metrics, and renders the Matplotlib visualization.
* `Survey_Results_UC.csv`: The raw dataset containing 96 survey responses across 60 questions.
* `Report.pdf`: The final 7-page analytical report detailing our methodology, structural findings, and ideological interpretations.
* `network-visualisation.png`: The generated network visualization exported from the script.

## Core Methodology

1. **Data Preprocessing:** Likert-scale responses are quantified (1 to 5). Empty rows and the ID column are dropped. Singular missing values are imputed to 'Neutral' (3) to preserve correlation integrity.
2. **Network Generation:** The network represents 60 survey questions as nodes. Edges are formed based on a Pearson correlation threshold of `r > 0.33`.
3. **Topological Analysis:** The script utilizes `NetworkX` to compute Degree (Predictors), Betweenness (Bridges), and Eigenvector (Influencers) centralities for all nodes.
4. **Decoupled Visualization:** A customized `spring_layout` in `Matplotlib` is used to force apart the highly dense core (Ethics/Environment) while mathematically arranging isolated nodes into an outer orbital ring.

## Prerequisites & Installation

The script requires Python 3.x and the following standard data science libraries. You can install the required dependencies via pip:

```bash
pip install pandas numpy networkx matplotlib
```

## Usage
Ensure the `Survey_Results_UC.csv` file is in the same directory as the script. Execute the Python file from your terminal:

```bash
python code.py
```

### Expected Output

1. The terminal will print the global network metrics (Total Nodes: 60, Total Edges: 450).
2. It will output a categorized breakdown of the top predictor, bridging, and influencer nodes for all four domains.
3. A Matplotlib window will open rendering the color-coded, 2D network visualization.

## Team Members

- Dataarnoor Singh Oberoi (2023102047) - Data Engineering & Network Definition
- Garima Mittal (2023102069) - Visual-Spatial Analysis & Typesetting
- Saarthak Sabharwal (2023102055) - Topological Analysis & Metric Synthesis
