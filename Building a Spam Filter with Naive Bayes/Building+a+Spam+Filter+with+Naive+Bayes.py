#!/usr/bin/env python
# coding: utf-8

# # Building a Spam Filter with Naive Bayes
# 
# In this guided project, we're going to study the practical side of the Naive Bayes algorithm by building a spam filter for SMS messages.
# 
# To classify messages as spam or non-spam, the computer:
# 
# 1. Learns how humans classify messages.
# 2. Uses that human knowledge to estimate probabilities for new messages — probabilities for spam and non-spam.
# 3. Classifies a new message based on these probability values — if the probability for spam is greater, then it classifies the message as spam. Otherwise, it classifies it as non-spam (if the two probability values are equal, then we may need a human to classify the message).

# ## Exploring the data

# In[1]:


# open the file
import pandas as pd
sms_data = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['Label', 'SMS'])
sms_data.head()


# In[2]:


sms_data['Label'].value_counts(normalize=True)


# On the previous screen, we read in the dataset and saw that about 87% of the messages are ham ("ham" means non-spam), and the remaining 13% are spam.

# ## Training and test dataset
# 
# To test the spam filter, we're first going to split our dataset into two categories:
# 
# * A training set (80%, 4458 messages), which we'll use to "train" the computer how to classify.
# * A test set (20%, 1114 messages), which we'll use to test how good the spam filter is with classifying new messages.

# In[3]:


sms_randomized = sms_data.sample(frac=1, random_state=1)

train_test_split = round(len(sms_data) * 0.8)
train = sms_randomized[:train_test_split].reset_index(drop=True)
test = sms_randomized[train_test_split:].reset_index(drop=True)


# In[4]:


# Find the percentage of spam and ham in both the training and the test set
print(train['Label'].value_counts(normalize=True))
print(test['Label'].value_counts(normalize=True))


# The percentages are similar to what we have in the full dataset.

# ## Teaching the algorithm to classify new messages

# In[5]:


# remove all punctuation and change all letters to lowercase
import re

train['SMS'] = train['SMS'].apply(lambda x: re.sub('\W', ' ', x))
train['SMS'] = train['SMS'].str.lower()

train.head(2)


# In[6]:


# Create a vocabulary for the messages in the training set

# split each message in SMS column at the space character
train['SMS'] = train['SMS'].str.split(" ")

vocabulary = []
for sms in train['SMS']:
    for word in sms:
        vocabulary.append(word)
        
vocabulary = list(set(vocabulary))


# In[7]:


word_counts_per_sms = {unique_word: [0] * len(train['SMS']) 
                       for unique_word in vocabulary}

for index, sms in enumerate(train['SMS']):
    for word in sms:
        word_counts_per_sms[word][index] += 1
        
word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.head()


# In[8]:


train_clean = pd.concat([train, word_counts], axis=1)
train_clean.head()


# ## Calculating the constants
# 
# The Naive Bayes algorithm will need to answer these two probability questions to be able to classify new messages:
# 
# $$
# P(Spam | w_1,w_2, ..., w_n) \propto P(Spam) \cdot \prod_{i=1}^{n}P(w_i|Spam)
# $$$$
# P(Ham | w_1,w_2, ..., w_n) \propto P(Ham) \cdot \prod_{i=1}^{n}P(w_i|Ham)
# $$
# 
# Also, to calculate P(wi|Spam) and P(wi|Ham) inside the formulas above, we'll need to use these equations:
# 
# $$
# P(w_i|Spam) = \frac{N_{w_i|Spam} + \alpha}{N_{Spam} + \alpha \cdot N_{Vocabulary}}
# $$$$
# P(w_i|Ham) = \frac{N_{w_i|Ham} + \alpha}{N_{Ham} + \alpha \cdot N_{Vocabulary}}
# $$
# 
# 
# Calculate some constants first:

# In[9]:


spam_messages = train_clean[train_clean['Label'] == 'spam']
ham_messages = train_clean[train_clean['Label'] == 'ham']

# PSpam/ PHam
p_spam = len(spam_messages) / len(train_clean)
p_ham = len(ham_messages) / len(train_clean)

# NSpam/NHam is equal to the number of words in all the spam messages
# it's not equal to the number of spam/ham messages, 
# and it's not equal to the total number of unique words in spam/ham messages.
n_spam = spam_messages['SMS'].apply(len).sum()
n_ham = spam_messages['SMS'].apply(len).sum()

# Nvocablary
n_vocabulary = len(vocabulary)

# Laplace smoothing
alpha = 1


# Calculate P(wi|Spam) and P(wi|Ham) parameters:

# In[10]:


parameters_spam = {unique_word: 0 for unique_word in vocabulary}
parameters_ham = {unique_word: 0 for unique_word in vocabulary}

for word in vocabulary:
    n_word_given_spam = spam_messages[word].sum()   # spam_messages already defined in a cell above
    p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha*n_vocabulary)
    parameters_spam[word] = p_word_given_spam
    
    n_word_given_ham = ham_messages[word].sum()   # ham_messages already defined in a cell above
    p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha*n_vocabulary)
    parameters_ham[word] = p_word_given_ham


# ## Creating the spam filter
# 
# The spam filter can be understood as a function that:
# 
# * Takes in as input a new message (w1, w2, ..., wn)
# * Calculates P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn)
# * Compares the values of P(Spam|w1, w2, ..., wn) and P(Ham|w1, w2, ..., wn), and:
#     * If P(Ham|w1, w2, ..., wn) > P(Spam|w1, w2, ..., wn), then the message is classified as ham.
#     * If P(Ham|w1, w2, ..., wn) < P(Spam|w1, w2, ..., wn), then the message is classified as spam.
#     * If P(Ham|w1, w2, ..., wn) = P(Spam|w1, w2, ..., wn), then the algorithm may request human help.

# In[11]:


def classify(message):

    message = re.sub('\W', ' ', message)
    message = message.lower()
    message = message.split()

    p_spam_given_message = p_spam
    p_ham_given_message = p_ham
    
    for word in message:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
            
        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]

    print('P(Spam|message):', p_spam_given_message)
    print('P(Ham|message):', p_ham_given_message)

    if p_ham_given_message > p_spam_given_message:
        print('Label: Ham')
    elif p_ham_given_message < p_spam_given_message:
        print('Label: Spam')
    else:
        print('Equal proabilities, have a human classify this!')


# In[12]:


# test classify
classify('WINNER!! This is the secret code to unlock the money: C3421.')


# In[13]:


classify('Sounds good, Tom, then see u there')


# ## Measure the accuracy of the spam filter

# In[14]:


def classify_test_set(message):

    message = re.sub('\W', ' ', message)
    message = message.lower()
    message = message.split()

    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for word in message:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]

        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]

    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_spam_given_message > p_ham_given_message:
        return 'spam'
    else:
        return 'needs human classification'


# In[15]:


test['predicted'] = test['SMS'].apply(classify_test_set)
test.head()


# In[16]:


correct = 0
total = test.shape[0]
    
for row in test.iterrows():
    row = row[1]
    if row['Label'] == row['predicted']:
        correct += 1
        
print('Correct:', correct)
print('Incorrect:', total - correct)
print('Accuracy:', correct/total)

