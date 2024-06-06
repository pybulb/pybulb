import pickle

class Packager:
    def package(self, obj):
        return pickle.dumps(obj)
    
    def unpackage(self, data):
        return pickle.loads(data)
