
<h3 align="center">Factors influence on NIM indicator in banking industry from 2012 to 2022</h3>

## ✏️ Data
I collected data from financial statemnets of 14 Vietnam commercial banks over the period of 11 years from 2012 to 2022

##  ✒️ Terms and formula
- **NIM:** Net interest margin = Net Revenue/Average interest-earning assets
- **CAR:** Capital Adequacy Ratio 
- **OCR:** Operating Cost Ratio = Operating cost/Average operating cost

## 📝 Table of Contents

- NIM over 11-year period in general
- NIM in the specific banks over the period
- Plots of some factors including NPL, CAR, OCR, GDP, CPI
- Correlation matrix between factors and NIM
- Models: Lasso and Random Forest Tree


## 1. NIM over 11-year period in general 

Pivot table and chart result:

In general, there was an uptrend in the NIM indicator for over 10-year period A plunge was seen in the year of 2013 at about 0.023 and continued keeping a low rate in 2014. After that, the rate increased gradually. If compared with the chart of CPI by year, we can see a significant decrease from 2012 to 2015. There was likely the relationship between CPI and NIM. However, NIM indicator was recovered sooner than CPI indicator Factly, Vietnam implemented a stricter monetary policy in order to control price in the period of 2013-2015.

## 2. NIM in the specific banks over the period 

Top 5 greatest NIM banks are Techcombank, VIB, VPbank, Vietcombank and Vietinbank. Some highlights for the fluctuation of NIM over the period was observed following:

Techcombank was seen in the uptrend and decrease gradually from 2017 to 2021 and decrease slightly in 2022 VPbank experienced the highest increase from 2014 to 2019 and then plunged and recovered until 2022 VIB was observed in the good state of NIM from 2016 to 2022 Although Vietcombank and Vietinbank had a quite low rate of NIM compared other banks, it can not conclude that these banks were not operating well. Because these two banks are two banks which has greatest amount of earning assets. Net revenue only includes amount of interest from lending. For example, Vietcombank gets a large earning from stock investment every year.


### 3. Plots of some factors including NPL, CAR, OCR, GDP, CPI 

Plot 3 section shows NPL of top banks during the period. In general, NPL ratio for all banks decreased. Especially in two countries including Techcombank and Vietcombank had both an significant decrease in this ratio. Techcombank completed well in the task of non-performing debts management from 2013 to 2020, and slightly increase after that. It was easily seen from the chart, VPbank had some issues with debt performance from 2020 to 2022. When seen in the chart of NIM, VPbank experienced a plunge in this period.

Plot 4 indicates that CAR of 5 banks fluctuated and did not show the exact trend over the period. In general, 5 banks kept the minimum of the ratio which is 8% according to Basel II. It is only VIB whose ratio increased considerably from 2015 to 2019 and Vietinbank from 2017 to 2022 broke the minimum rate, which was under 8%.

Plot 5 shows the distribution of independent variables. NPL ratio was mostly between 0.00 to 0.03, which complied with the norm ratio of 0.03 and CAR was below 0.02, which was a good ratio. OCR ratio was below 0.04 mostly. GDP and CPI was majorly between 6-8% and 2-4% respectively.

### 4. Correlation matrix between factors and NIM

Correlation matrix shows the correlation between independent variables and dependent variable (NIM). There are only two variables which comparatively correlate with NIM: OCR and NPL. OCR has a positive relationship with NIM while NPL has a negative relationship with NIM. There is a significant relationship between NPL and CPI compared to other ones. This was a positive correlation which means these two ratio had the same trend over the period.


## 5. Models: Lasso and Random Forest Tree

I implemented two regressions: Lasso and Random Forest Regression to figure out the relationship between independent variables and dependent variable Lasso gave the coefficents which are weak and the overall score or R-squares is only about 25%. MSE indicate is approximately 0 in both training and test data. So, overfitting exists in the model.

Using random forest regression, the importance variables which could affect on the NIM are OCR and NPL.