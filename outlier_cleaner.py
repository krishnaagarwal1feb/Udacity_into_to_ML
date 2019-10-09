#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    cleaned_data = []
    import math
    ### your code goes here
    li = []
    sz = len(predictions)
    for x in range(sz):
        li.extend( abs(predictions[x] - net_worths[x]) )
    li.sort()
    keep = int(math.floor(len(li)*0.9))
    error = li[:keep]
    for x in range(sz):
        err =  abs(predictions[x] - net_worths[x])
        if err in error:
            cleaned_data.append([ages[x], net_worths[x], err])
    return cleaned_data
