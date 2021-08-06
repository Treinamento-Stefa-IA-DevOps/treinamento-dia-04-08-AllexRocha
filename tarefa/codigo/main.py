import pickle
from fastapi import FastAPI

app = FastAPI()
@app.post('/model')
## Coloque seu codigo na função abaixo
async def titanic(Sex: int, Age: float, Lifeboat: int, Pclass: int):
    with open('model/Titanic.pkl', 'rb') as fid:
        titanic = pickle.load(fid)

    previsao = titanic.predict([[Sex, Age, Lifeboat, Pclass]]).tolist()
    
    surv = previsao[0]

    try:
        return {
            'survived': bool(surv),
            'status': 200,
            'message': 'Predição bem sucedida!',
        }
    except Exception:
        return {
            'message': 'Erro no servidor'
        }


#@app.get('/model')
#async def get():
#    return {
#       'hello': 'test'
#    }
