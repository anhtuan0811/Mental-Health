# 👨🏼‍⚕️ Mental Health Prediction


# :books: Table of Contents <!-- omit in toc -->

- [:briefcase: Case Study and Requirement](#case-study-and-requirement)
- [:bookmark_tabs: Example Datasets](#bookmark_tabsexample-datasets)
- [🔎 Explore data and test model](#explore-data-and-test-model)
- [📃 What can you practice with this case study?](#-what-can-you-practice-with-this-case-study)

---

# Case Study and Requirement

This dataset is from a 2014 survey that measures attitudes towards mental health and frequency of mental health disorders in the tech workplace.

### ❓ Question
Can you predict whether a patient should be treated of his/her mental illness or not according to the values obtained in the dataset?

---

# :bookmark_tabs:Example Datasets

<details><summary> 👆🏼 Click to expand Dataset information </summary>

- Timestamp
- Age
- Gender
- Country
- state: If you live in the United States, which state or territory do you live in?
- self_employed: Are you self-employed?
- family_history: Do you have a family history of mental illness?
- treatment: Have you sought treatment for a mental health condition?
- work_interfere: If you have a mental health condition, do you feel that it interferes with your work?
- no_employees: How many employees does your company or organization have?
- remote_work: Do you work remotely (outside of an office) at least 50% of the time?
- tech_company: Is your employer primarily a tech company/organization?
- benefits: Does your employer provide mental health benefits?
- care_options: Do you know the options for mental health care your employer provides?
- wellness_program: Has your employer ever discussed mental health as part of an employee wellness program?
- seek_help: Does your employer provide resources to learn more about mental health issues and how to seek help?
- anonymity: Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?
- leave: How easy is it for you to take medical leave for a mental health condition?
- mentalhealthconsequence: Do you think that discussing a mental health issue with your employer would have negative consequences?
- physhealthconsequence: Do you think that discussing a physical health issue with your employer would have negative consequences?
- coworkers: Would you be willing to discuss a mental health issue with your coworkers?
- supervisor: Would you be willing to discuss a mental health issue with your direct supervisor(s)?
- mentalhealthinterview: Would you bring up a mental health issue with a potential employer in an interview?
- physhealthinterview: Would you bring up a physical health issue with a potential employer in an interview?
- mentalvsphysical: Do you feel that your employer takes mental health as seriously as physical health?
- obs_consequence: Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?
- comments: Any additional notes or comments

</details>

<details><summary> 👆🏼 Click to expand Dataset sample rows </summary>

<div align="center">

**Table** 

<div align="center">
First 10 rows

|Timestamp|Age|	Gender|	Country|	state|	self_employed|	family_history|	treatment|	work_interfere|	no_employees|	remote_work|	tech_company|	benefits|	care_options|	wellness_program|	seek_help|	anonymity|	leave|	mental_health_consequence|	phys_health_consequence|	coworkers|	supervisor|	mental_health_interview|	phys_health_interview|	mental_vs_physical|	obs_consequence|	comments|
|:----|:-----|:----|:----|:----|:----|:----|:----|:----|:-----|:----|:----|:----|:----|:----|:----|:----|:-----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
2014-08-27 11:29:31|	37|	Female|	United States|	IL|	NA|	No|	Yes|	Often|	6-25|	No|	Yes|	Yes|	Not sure|	No|	Yes|	Yes|	Somewhat easy|	No|	No|	Some of them|	Yes|	No|	Maybe|	Yes|	No|	NA|
2014-08-27 11:29:37|	44|	M|	United States|	IN|	NA|	No|	No|	Rarely|	More than 1000|	No|	No|	Don't know|	No|	Don't know|	Don't know|	Don't know|	Don't know|	Maybe|	No|	No|	No|	No|	No|	Don't know|	No|	NA|
2014-08-27 11:29:44|	32|	Male|	Canada|	NA|	NA|	No|	No|	Rarely|	6-25|	No|	Yes|	No|	No|	No|	No|	Don't know|	Somewhat difficult|	No|	No|	Yes|	Yes|	Yes|	Yes|	No|	No|	NA|
2014-08-27 11:29:46|	31|	Male|	United Kingdom|	NA	|NA	|Yes	|Yes	|Often	|26-100	|No	|Yes	|No	|Yes	|No	|No	|No	|Somewhat difficult	|Yes	|Yes	|Some of them	|No	|Maybe	|Maybe	|No	|Yes	|NA|
2014-08-27 11:30:22|	31|	Male|	United States|	TX	|NA|	No|	No|	Never|	100-500|	Yes|	Yes|	Yes|	No|	Don't know|	Don't know|	Don't know|	Don't know|	No|	No|	Some of them|	Yes	|Yes	|Yes	|Don't know	|No	|NA|
2014-08-27 11:31:22|	33|	Male|	United States|	TN|	NA|	Yes|	No|	Sometimes|	6-25|	No|	Yes|	Yes|	Not sure|	No|	Don't know|	Don't know|	Don't know|	No|	No|	Yes|	Yes|	No|	Maybe|	Don't know	|No|	NA|
2014-08-27 11:31:50|	35|	Female|	United States|	MI|	NA|	Yes|	Yes|	Sometimes|	1-5|	Yes|	Yes|	No|	No|	No|	No|	No|	Somewhat difficult|	Maybe|	Maybe|	Some of them|	No|	No|	No|	Don't know|	No|	NA|
2014-08-27 11:32:05|	39|	M|	Canada|	NA|	NA|	No|	No|	Never|	1-5|	Yes|	Yes|	No|	Yes|	No|	No	|Yes|	Don't know|	No|	No|	No|	No|	No|	No|	No|	No|	NA|
2014-08-27 11:32:39|	42|	Female|	United States|	IL	|NA	|Yes	|Yes	|Sometimes	|100-500	|No	|Yes	|Yes	|Yes	|No	|No	|No	|Very difficult	|Maybe	|No	|Yes	|Yes	|No	|Maybe	|No	|No	|NA|
2014-08-27 11:32:43|	23|	Male|	Canada|	NA|	NA|	No|	No|	Never|	26-100|	No|	Yes|	Don't know|	No|	Don't know|	Don't know|	Don't know|	Don't know|	No|	No|	Yes|	Yes|	Maybe|	Maybe|	Yes|	No|	NA|

</div>
</div>

</details>

---
## 🔎  Explore data and test model

### The Process is following - [Code & Presentation](https://github.com/beto1810/Predictors-of-mental-health-illness/blob/main/Explore%20data%20and%20test%20model.md#1%EF%B8%8F%E2%83%A3-explore-data-analysis) or [Only Code](https://github.com/beto1810/Predictors-of-mental-health-illness/blob/main/File/Final_Mindx_De1%20(2).ipynb)

- Import Library and dataset
- Explore data
- Preprocessing - Encoding
- Covariance Matrix
- Relationship Charts
- Scaling & Fitting
- Tuning
- Evaluate Model
- Success method plt
- Creating predictions on test set


---

# 🧾 What can you practice with this case study?
- Python
  - pandas, numpy,matplotlib,seaborn, scpipy.
  - Cleaning, check Null values, transforming.
  - Running model,Scaling model, Fiting model, Testing model. 
  - Def function.
  

