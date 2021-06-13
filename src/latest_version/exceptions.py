class InvalidFileTypeException(BaseException):
    pass


class LabelColumnNotFoundException(BaseException):
    pass


class InvalidPreprocessParametersException(BaseException):
    pass


class FeatureMethodNotFoundException(BaseException):
    pass


class NullIndexException(BaseException):
    def __init__(self, null_index):
        self.null_index = null_index
        self.notify()

    def notify(self):
        print(f"Row: {self.null_index[0][0]} Col: {self.null_index[1][0]}" )



class LayerInfoNotFoundException(BaseException):
    pass


class InvalidSagemakerParameterException(BaseException):
    pass