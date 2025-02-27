from tabulate import tabulate
import logging
def print_args_table(args):
    """Print the arguments as a formatted table using tabulate.
    
    Args:
        args: The parsed command-line arguments.
    """
    table_data = []
    for arg in list(vars(args).keys()):
        value = getattr(args, arg)
        table_data.append([arg, value, type(value).__name__])
    print(tabulate(table_data, headers=["Parameter", "Value", "Type"], tablefmt = 'psql'))
    print()

def info_args_table(args):
    """Print the arguments as a formatted table using tabulate.
    
    Args:
        args: The parsed command-line arguments.
    """
    table_data = []
    for arg in list(vars(args).keys()):
        value = getattr(args, arg)
        table_data.append([arg, value, type(value).__name__])
    logging.info(tabulate(table_data, headers=["Parameter", "Value", "Type"], tablefmt = 'psql'))
    logging.info(' ')