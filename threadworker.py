import logging #report bot events   
import telegram
import sys
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, filters
import os
import time
from queue import PriorityQueue
from sd import StableDiffusion
import time
import logging
import threading
from sd import *
from myqueue import *


class hiloEnviar:

    
    
    def __init__(self,cola,emptyCondition,sd,dic):
        self.cola = cola
        self.emptyCondition= emptyCondition
        self.sd=sd
        self.dic=dic
    
    def start(self):
        while True:
            print("hola")
            self.emptyCondition.acquire()
            isnotEmpty = self.cola.tamano() > 0
            print(f"{isnotEmpty}")
            while not isnotEmpty:
                 print("dormidito")
                 self.emptyCondition.wait()
                 isnotEmpty = self.cola.tamano() > 0
                 
            print("entro")

            tupla = self.cola.sacarRes()
            print(f"{tupla}")
            self.emptyCondition.release()
            # proceso tupla....... (llamar stable diffusion)
            bot=tupla[1]
            chat_Id=tupla[2]
            user_Id=tupla[3]  
            msg=tupla[0] 
            self.sd.txtToImg(msg,self.dic,user_Id)
            
            fichero = open('url.txt')
            img_name=f"imagenes/{fichero.read()}"    
          
            
            print("acabado")
            tupla[4](bot,chat_Id,img_name)
            
