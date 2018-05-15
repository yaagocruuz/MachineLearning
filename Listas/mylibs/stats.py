from math import sqrt

def somar(x):
    soma = 0
    for v in x:
        soma = soma + float(v)
    return soma

def mean(x):
    qtd_elementos = len(x)
    media = somar(x) / float(qtd_elementos)
    return media
 
def var(y):
    media = mean(y)
    soma = 0
    variancia = 0
 
    for valor in y:
        soma += (float(valor) - media) ** 2
   
    variancia = soma / float(len(y))
    return variancia

def stddev(x):
    desvio_padrao = sqrt(var(x))
    return desvio_padrao

