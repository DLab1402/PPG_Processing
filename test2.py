from abc import ABC, abstractmethod

class MyBaseClass(ABC):
    @abstractmethod

    def my_abstract_function(self,**kwargs):
        pass

    def run(self):
        self.my_abstract_function()

class MyInheritingClass(MyBaseClass):
    def my_abstract_function(self, arg1, arg2, arg3):
        # Implement the function with an additional argument
        result = arg1 + arg2 + arg3
        print(result)

# Example usage
instance = MyInheritingClass()
instance.run()