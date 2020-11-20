# starbucks-capstone
Udacity data science nanodegree capstone project.

## Table of Contents
- [Project overview](#overview)
- [Content](#content)
- [Installation](#installation)
- [Analysis](#analysis)

### Project overview <a name="overview"></a>
The data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, 
Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual 
offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks.

### Content <a name="content"></a>
The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record

### Installation <a name="installation"></a>
1. Make sure that python is installed by issuing this command on the terminal:
```sh
$ python --version
```
2. Also you could follow the steps for installing pip, if not installed, over [here][pip-install] 
3. (optional) create a separate python env. Here is how to create [virtual env][env-install]
4. Execute this command: _(This should download all necessary packages for running this project)_.
```sh
$ pip install requirements.txt
```
5. The project contains a jupyter file, so it needs to be run using jupyter notebook. This can be done through anaconda, or just executing this command: (This will provide a web interface for jupyter notebook for running jupyter files).  
```sh
$ jupyter notebook
```

### Analysis <a name="analysis"></a>
Simple Analysis for the problem is provider in this [article][medium-post]

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [pip-install]: <https://pip.pypa.io/en/stable/installing/>
   [env-install]: <https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/>
   [medium-post]: <https://mkodary.medium.com/starbucks-capstone-challenge-fe127aee2081>
