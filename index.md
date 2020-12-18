# Improving the performance and explainability of civil war onset predictive models

## 1. Introduction

> Both the report/data story and the video pitch will be evaluated by people who are not closely familiar with the paper you are working on. It is important to first explain the main ideas from the paper, and how exactly the creative extension builds on it.

In this project we extend the civilwar paper. We attempt to answer three questions

## 2. Improving the Performance

You can use the [editor on GitHub](https://github.com/TiannanSha/ada-data-story/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### 2.1 Neural Networks for predicting civil war
![Image](/images/arch_NN90.png)

### 2.1 Synthetic Minority Over-sampling Technique

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

## 3 Causal Inference of the GDP related features

### 3.1 Background

In the civil war paper, it says that:

> Since Random Forests does cross-validation internally through the use of OOB observations, we can use these OOB observations to determine some aspects of causality.
>
> ...
>
> Figure 4 shows the mean decrease in a measure of predictive accuracy, called the Gini Score, for the top twenty variables that Random Forests picked.
>
> ...
>
> Variables with a greater mean decrease in the Gini Score more accurately predict the true class of observation i when i is OOB.
>
> ...
>
> The best predictor of civil war onset is national poverty as measured by both the growth rate of national gross domestic product (GDP) and GDP per capita. This reinforces the results of nearly every examination into the causes of civil war onset published in the past decade.



However, it must be very cautious while talking about the causality. In most of statistic models, it can only get the result of correlation, but not the causality.  So when we see that the authors of the paper use Random Forest model to get "some aspects of causality", we are worried about the result, because the Random Forest model seems not have ability to do the causal inference. Although the model have the OOB observations to do the evaluation and comparison, but the comparison is based on a very imbalance dataset. That means the comparison can't get the result of causality.

In our work, we will do the causal inference to learn if the value of GDP growth can really lead to civil war onset by using the same dataset and features as the Random Forest model.  

### Causality

One of the most widely used model of causal inference is the Neyman-Rubin model.  In a basic Neyman-Rubin model, if we want to know the causal effect of a feature T, we will assign the samples into two groups based on the value of T: 

* Treatment group: T = 1
* Control group: T = 0

Then, the potential output of the two groups, Y|T=1 and Y|T=0 can be compared and show the causal effect of feature T: E(Y|T=1) - E(Y|T=0)

However, this idea is based on a assumption that the other features of the Treatment group and Control group are similar, they only have difference in the feature T. It's very difficult to hold this assumption for the observation data because we can't determine the distribution of the covariates by randomization.  So if it's possible to use this causality model even in the observation study?

The answer is yes. What we need is the tool called propensity scores. A propensity score is the conditional probability that a subject receives “treatment” given the subject’s observed covariates: prop_score = P(T = 1|X)

The goal of propensity scoring is to  balance observed covariates between subjects in control and treatment study groups. The researchers have proven that when the propensity score is balanced in the treatment and control group, we can get the causal effect of feature T properly. So we will start with find a good propensity scores and balance the propensity scores of the two groups.

After we get the propensity scores, we can then do the "Propensity Score Matching". By doing this, each pair of samples has similar Propensity Scores so we can compare their outputs. 

Our architecture to do the causal inference is:

![](images/causal1.PNG)

### Treatment

Because we want to get the causal effect of feature GDP Growth, we must find a good way to assign the samples into the Treatment Group and Control Group. If the Treatment T is a binary value, the samples can be divided based on the binary value of T. However, the GDP Growth is continuous  float numbers, and the continues treatment problem is much more complex than the binary case. So our basic idea is to discretize the GDP Growth:

![](images/causal2.png)

We divide the samples into three parts based on the quantile of the GDP Growth. In order to make sure the GDP Growth of the Treatment Group and Control Group have a "gap", we choose to drop the "middle" samples.

![](images/causal4.png)

When the alpha is too big, the difference between the Treatment Group and Control Group will close to 0 and there is no gap between them. However when the alpha is too small, the sample size of each group will be small, and we can only use a little part of the dataset.

![](images/causal3.png)

 After tuning the alpha,  we choose to let alpha=0.3 and we can use 60% of the samples.

What's more, from the density plot, we can see that the positive and negative samples has the similar distributions in each group.



### Covariates

Based on the Random Forest Model, we can select some features as covariates. Now let's show that these covariates are imbalanced in Treatment and Control Group, by using the t-test of the mean of two independent samples:

![](images/causal5.png)

Most of the features are imbalanced in the Treatment Group and Control Group. So it's essential to use a propensity scores to balance the samples.



### Propensity score

We train a model to predict if the sample is in the treatment group by using the the covariates. The probability of the prediction can be regard as the  propensity score. We try three different models to get the propensity score.

![](images/causal6.png)

Based on the result above, we choose to use the Logistic Regression model to generate the propensity score. Then we can do the matching.



### Matching

We can match the samples by using a greedy algorithm:

![](images/causal7.png)

1. Drop some samples to make sure the two groups have the same range of propensity score.

2. t = min_propensity_score(Treatment), the sample is s_t

   c = min_propensity_score(Control), the sample is s_c

3. Compare t and c:
   1. t - c > delta: drop s_c
   2. t - c < -delta: drop s_t
   3. otherwise: match (s_c, s_t) and drop them
4. If there is no sample in Treatment Group or Control Group, return all the matches. Otherwise go back to step 2

 By running this algorithm, we can get the differences of the propensity scores in each pair:

![](images/causal8.png)

We can find when the bound become smaller, the number of pairs becomes smaller. But when the bound become bigger, the quality of matches become worse. Finally we choose bound = 0.0002.

After the matching, we can test the balance of the covariates again and get the result:

![](images/causal9.png)

That means all covariates are balanced.



### Mean causal effect


### Train Random Forest model with the matching data

Finally, we choose to train a Random Forest model by using the matching data only. The importance of the features is:

![](images/causal10.png)

That means even the gdpgrowth have no significant causal effect, it's still the most important feature.

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/TiannanSha/ada-data-story/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
