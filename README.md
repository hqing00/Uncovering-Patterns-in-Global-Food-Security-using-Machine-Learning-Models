# Predictive Modeling and Country Clustering from Global Food Security Indicators
This project was developed as part of a coursework assignment in Data Analytics (WQD7003)

## Description
Developed predictive models and performed country clustering using global food security indicators to uncover patterns in food access, availability, and stability across regions. The project aimed to deliver actionable insights for addressing food insecurity challenges.

## Problem Statement
Food insecurity remains a critical global issue, with significant variation in access, availability, and supply stability across countries. This project aims to identify patterns and group countries based on food security indicators to support targeted policy interventions and predictive risk monitoring.

## Data
There are 3 datasets was obtained:
1. FAOSTAT FS: https://www.fao.org/faostat/en/#data/FS
2. FAOSTAT QCL: https://www.fao.org/faostat/en/#data/QCL
3. IMF Climate Data: https://climatedata.imf.org/

## Insights
- Cluster 0 (Red): countries facing multiple food security challenges with significantly inconsistent supply patterns.
- Cluster 1 (Blue): countries with adequate protein supply, low undernourishment, low food insecurity, and consistent food supply.
- Cluster 2 (Green): generally food secure but experience supply variability (possibly due to climate, trade dependencies, or economic fluctuations).
- Cluster 3 (Orange): consistently poor food security indicators but though inadequate supply patterns.
- LightGBM and Random Forest were chosen for their robustness and ability to perform well even on relatively small datasets (~1.5k records).

## Recommendations
- Use cluster insights to prioritize intervention, allocate resources and form regional partnership.
- Implement early warning systems or real-time dashboard to monitor and mitigate supply shocks.

## Team Members
1. Liew Cai Tong
2. Lam Jun Yan
3. Chew Hong Ern
4. Yap Hui Qing
5. Yeoh Li Ting
