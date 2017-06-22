# Print iterations progress
def print_progress_bar(iteration: int, total: int, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


# noinspection PyTypeChecker
def main():
    """
    Sample usage
    """
    from time import sleep

    # make a list
    items = list(range(0, 57))
    i = 0
    l = len(items)

    # Initial call to print 0% progress
    print_progress_bar(i, l, prefix='Progress:', suffix='Complete', length=50)
    for i in range(57):
        # Do stuff...
        sleep(0.1)
        # Update Progress Bar
        print_progress_bar(i, l, prefix='Progress:', suffix='Complete', length=50)


if __name__ == "__main__":
    main()
