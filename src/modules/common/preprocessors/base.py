from abc import abstractmethod


class PreProcessorBase:
    @abstractmethod
    def run(self):
        pass