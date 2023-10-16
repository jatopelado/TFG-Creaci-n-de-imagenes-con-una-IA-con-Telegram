import dill

class Settings:
    def __init__(self):
        self.dic=self.cargar_mapa()

    def claves(self):
       return self.dic.keys()
    
    def añadiropciones(self,userId,opciones):
        self.dic.update({userId : opciones})

    def obteneropciones(self,userId):
        return self.dic[userId]
    
    def obtenerdic(self):
        return self.dic

    def guardar_mapa(self,):
        with open("settings.pickle", 'wb') as archivo:
            dill.dump(self.dic, archivo)

    def cargar_mapa(self):
        with open("settings.pickle", 'rb') as archivo:
            diccionario = dill.load(archivo)
        return diccionario

    

