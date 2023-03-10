"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    [makes noise 0 to negate randomness to always go ideal path]
    """

    answerDiscount = 0.9
    answerNoise = 0

    return answerDiscount, answerNoise

def question3a():
    """
    [lower gamma makes agent prefer immediate reward over future reward
    lower noise makes agent can be controlled easily
    negative living reward makes agent prefer to reach the goal as soon as possible
    which is the exit 1]
    """

    answerDiscount = 0.4
    answerNoise = 0
    answerLivingReward = -5.0

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    [lower gamma makes agent prefer immediate reward over future reward
    a little noise makes d]
    """

    answerDiscount = 0.2
    answerNoise = 0.1
    answerLivingReward = 0.1

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    [higher discount prefer higher exit, 0 living reward will not prefer to exit later.]
    """

    answerDiscount = 0.7
    answerNoise = 0.0
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    [default case works here.]
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    [makes really low living reward makes exit as soon as possible]
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -100.0

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    [impossible to reach exit 2 since the ideal way to reach exit 2 is randomly choice through the
    grid which takes average 4^5 exploration to reach exit 2. This is way more than 50 iterations]
    """
    return None

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
