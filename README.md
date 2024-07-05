# Speed Dating Experiment
## Technion - Israel Institute of Technology<br/>Department: The school of continuing education<br/>Course: Technion Certified Data Scientist

**Course project:** Exploratory Data Analysis of the dataset "Speed Dating Data" with Python.<br/>
**Dataset:** [Speed Dating Data](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment).<br/>
**Programming Language:** Python 3.10.<br/>
**Software:** Spyder IDE 5.4.1, Jupiter Notebook<br/>

**Dataset description:**<br/>
Data was gathered from participants in experimental speed dating events from 2002-2004. During the events, the attendees would have a four minute "first date" with every other participant of the opposite sex. At the end of their four minutes, participants were asked if they would like to see their date again. They were also asked to rate their date on six attributes: Attractiveness, Sincerity, Intelligence, Fun, Ambition, and Shared Interests.<br/>


**Task: In each of the described sections, present the data in the most appropriate and useful way in your opinion.**<br/>
1. Meta analysis
    * Number of samples, features
    * Dtypes
    * Missing values statistics (not imputation)
2. Fields analysis
    * About the people (choose 3): gender, age, race, field of study, career, activities interests
    * Dating: goal, date, go-out
    * Bi-variate analysis: (choose 3) “me-other” features: e.g. same race, career 1 & 2, shared activities, dating goal etc.
3. Dating (analyze each question separately for each gender)<br/>
    * What’s important for you (what you look for in the opposite sex)<br/>
        * During Sign-up questionnaire<br/>
        * During Half-way<br/>
        * Compare between both<br/>
    * What’s important for others of the same gender? (what you think MOST of your fellow men/women look for in the opposite sex)
    * What’s important for others of the opposite gender? (what do you think the opposite sex looks for in a date)
4. Me vs. others (bar/histogram)
    * How do I perceive myself? (how do you think you measure up?)
    * How do I think others perceive me? (how do you think others perceive you?)
    * How do others actually perceive me?
        * Define this based on dating partners' feedback.
        * Choose how to aggregate this feedback per participant based on his/her dating partners,<br/>
           and how to aggregate for the whole dataset, per gender.
        * Present both the values and the agreement score per feature for each gender.
           For example, how do men in general perceive their women partners' fun/intelligence?<br/>
           Is there a clear agreement about one’s attractiveness/intelligence etc.)?
    * How many do you expect will be interested in dating you vs. actual (agg. of dec_o)?
5. Effects of experiment setup/surroundings (% match by:)
   * Date order
   * Station number (is there any “lucky table”?)
   * Waves<br/>
6. Towards predicting a decision (correlations analysis)
   * Divide features to me/other/other-on-me data
   * My/other/other-on-me data vs. my/other target (decision/decision_o)
   * Which features are (not) important?<br/>

**Project files:**<br/>
* *Speed Dating Data - EDA.ipynb* - Exploratory Data Analysis of the dataset "Speed Dating Data".<br/>
* *Speed Dating Data Key.doc* - The dataset description.<br/>
* *Speed Dating Data.csv* - The dataset "Speed Dating Data".<br/>
