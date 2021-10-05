import numpy as np
import sys

class VeritasError():
    """Helper class that holds all the error messages."""
    def __init__(self):
        """
        Instance attributes
        -----------------------
        queue : list
               Holds the messages for errors
        """
        self.queue = list()

    def push(self, error_type, **kwargs):
        """
        Saves the errors to a template message depending on their error type to a list

        Parameters
        --------------
        error_type : str
                Each error_type corresponds to a message template

        Other parameters
        --------------
        **kwargs : dict
                Various arguments for the error messages

        Returns
        ---------------
        Appends error messages to the queue
        """
        if error_type == 'value_error':
            var_name = kwargs['var_name']
            expected = kwargs['expected']
            given = kwargs['given']
            function_name = kwargs['function_name']
            errMsg = "{}: given {}, expected {}".format(var_name, given, expected)
            self.queue.append([error_type, errMsg, function_name])

        if error_type == "value_error_compare":
            var_name_a = kwargs['var_name_a']
            var_name_b = kwargs['var_name_b']
            function_name = kwargs['function_name']
            errMsg = "{} cannot be less than {}".format(var_name_a, var_name_b)
            self.queue.append([error_type, errMsg, function_name])

        if error_type == 'conflict_error':
            var_name_a = kwargs['var_name_a']
            # var_name_b = kwargs['var_name_b']
            some_string = kwargs['some_string']
            value = kwargs['value']
            function_name = kwargs['function_name']
            errMsg =  "{}: {} {}".format(var_name_a, some_string, value)
            self.queue.append([error_type, errMsg, function_name])

        if error_type == 'type_error':
            var_name = kwargs['var_name']
            expected = kwargs['expected']
            given = kwargs['given']
            function_name = kwargs['function_name']
            errMsg = "{}: given {}, expected {}".format(var_name, given, expected)
            self.queue.append([error_type, errMsg, function_name])

        if error_type == 'length_error':
            var_name = kwargs['var_name']
            expected = kwargs['expected']
            given = kwargs['given']
            function_name = kwargs['function_name']
            errMsg = "{}: given length {}, expected length {}".format(var_name, given, expected)
            self.queue.append([error_type, errMsg, function_name])

        if error_type == 'column_value_error':
            var_name = kwargs['var_name']
            given = kwargs['expected_range']
            expected = kwargs['col']
            function_name = kwargs['function_name']
            errMsg = "{}:  expected {} at {}".format(var_name, given, expected)
            self.queue.append([error_type, errMsg, function_name])

    def pop(self):
        """
        Prints error messages and exits the programme.
        """
        msgs = ''
        if len(self.queue) > 0:
            for i in self.queue:
                msgs += "[{}]: {} at {}()\n".format(i[0], i[1], i[2])
            self.queue = list()
            sys.exit(msgs)
        # if msg_type != '' and msg_value == '':
        #     raise TypeError(msg_type)
        # elif msg_type == '' and msg_value != '':
        #     raise ValueError(msg_value)
        # elif msg_type == '' and msg_value == '':
        #     print("No errors")
        # else:
        #     try:
        #         raise (msg_type)
        #     except:
        #         raise ValueError(msg_value)
