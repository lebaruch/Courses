#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    ### your code goes here
    errors = (net_worths-predictions)**2
    threshold = numpy.percentile(numpy.absolute(errors), 90)
    #cleaned_data = zip(ages, net_worths, errors)
    #cleaned_data = sorted(cleaned_data, key=lambda x:x[2][0], reverse=True)
    #limit = int(len(net_worths)*0.1)
    cleaned_data = [(age, net_worth, error) for age, net_worth, error in zip(ages, net_worths, errors) if abs(error) <= threshold]

    return cleaned_data
