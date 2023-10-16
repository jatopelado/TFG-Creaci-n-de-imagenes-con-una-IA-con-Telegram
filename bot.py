
import logging #report bot events   
import telegram
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, filters
from sd import StableDiffusion
import time
import logging
import threading
from sd import *
from threadworker import *
import torch
from myqueue import *
from settings import*
import pickle




colaenv=Cola()
DicID=Settings()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s," 
)
logger = logging.getLogger()



TOKEN="6170568695:AAHhQd__FaVSm5c7072oxJEtCFraiBIMMUQ"

emptyCondition = threading.Condition()


def enviarfoto(bot,chatid,img_name):
  bot.sendPhoto(chat_id=chatid,photo=open(f"{img_name}", 'rb'))

def txt2img(update, context):
    userId = update.message.from_user
    i=1
    bot = context.bot
    chatId = update.message.chat_id
    args = " ".join(context.args)
    ids=DicID.claves()
    if not userId in ids:
        opciones={}
        DicID.añadiropciones(userId, opciones)
    if len(args) == 0: #if there is not extra info in command
            print("hola3")
            logger.info(f'no se ha establecido mensaje')
            bot.sendMessage(chat_id=chatId, text=f'no has escrito nada')
        
    else: #there is extra info
        emptyCondition.acquire()
        aux=colaenv.copiar()

        print("holaprincipal")
        while  aux.tamano() > 0:
            copia=aux.sacar
            print("copia")
            print(copia)
            print("hola")
            comparacion=copia[1]
            
            if comparacion[2] == chatId:
                print('entroalif')
                i+=1
            
        tupla=(args,bot,chatId,userId,enviarfoto)
        colaenv.añadir((i,tupla))
        print("metido")
        colaenv.ordenar
        emptyCondition.notify()
        emptyCondition.release()
        
    
    bot.sendMessage(chat_id=chatId, text=f'preparando mensaje')
    #envmensage()
  
      

def bienvenida(update, context):
    bot = context.bot
    chatId = update.message.chat_id
    bot.sendMessage(chat_id=chatId, text=f'Usa /crear seguido de un texto para usarme\nEjemplo: \n/crear a dog aeting a mango\nUsa /modoopciones para establecer tus opciones\nLas opciones disponibles son ancho(w), alto(h) y la semilla (seed)\nEjemplo:\n/modopciones seed:4, w:256, h:256\nY usa /help para volver a mostrar este mensje' )

def settings(update , context):
    print(update.message)      
    
    chatId = update.message.chat_id
    print(chatId)
    
    userId = update.message.from_user
    
    bot = context.bot
    args = " ".join(context.args)  
    listaopciones=args.split(',')
    print(listaopciones)
    ids=DicID.claves()
    if userId in ids:
        opciones=DicID.obteneropciones(userId)
    else:
        opciones={}
    for i in listaopciones:
      x=i.split(':')
      clave=x[0].strip()
      valor=int(x[1].strip())
      if clave =="w" or clave =="h":
          if valor<256:
            valor=256
          if 256<valor<512:
              valor=256
          if valor>512:
            valor=512
      opciones.update({clave : valor})
      
    DicID.añadiropciones(userId,opciones)
    DicID.guardar_mapa()
    bot.sendMessage(chat_id=chatId, text=f'opciones actualizadas')
    

    
          
                


if __name__ == "__main__":
  
    
  for i in range(torch.cuda.device_count()):
    sd1=StableDiffusion(i)
    hilo=hiloEnviar(colaenv,emptyCondition,sd1,DicID)
    thread = threading.Thread(target=hilo.start)
    thread.start()
    
  
  
    
    #información del bot
  myBot = telegram.Bot(token= TOKEN)       


#conectar el updater con el bot
  updater = Updater(myBot.token, use_context=True)

#despachar informacion
  dp = updater.dispatcher

#manejo de comandos
  dp.add_handler(CommandHandler("crear", txt2img,pass_args=True))
  dp.add_handler(CommandHandler("start", bienvenida))
  dp.add_handler(CommandHandler("help", bienvenida))
  dp.add_handler(CommandHandler("modopciones", settings,pass_args=True))
  
  updater.start_polling()

  updater.idle()


   




