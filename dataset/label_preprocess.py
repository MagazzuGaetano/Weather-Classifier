
def get_label(classname):
    label = 0
    if classname == "no weather degradation":
        label = 0
    elif classname == "fog":
        label = 1
    elif classname == "rain":
        label = 2
    elif classname == "snow":
        label = 3
    else:
        print('invalid gt!!!')
    return label
