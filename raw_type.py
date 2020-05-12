from abc import abstractclassmethod


class raw_type():
    @abstractclassmethod
    def read_data(self, path):
        pass

    @abstractclassmethod
    def split(self):
        pass

    @abstractclassmethod
    def annotation(self):
        pass

    @abstractclassmethod
    def correct(self):
        pass

    @abstractclassmethod
    def tensecond(self):
        pass

    @abstractclassmethod
    def filter_data(self):
        pass

    def process(self, path):
        self.read_data(path)
        self.filter_data()
        self.annotation()
        self.split()
        self.tensecond()
        # self.filter_data()
        self.correct()
