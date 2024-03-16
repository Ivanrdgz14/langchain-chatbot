![Banner](./LangChain%20Banner.webp)

## Langchain

Langchain es una poderosa biblioteca diseñada para transformar la interacción y el flujo de trabajo con modelos de lenguaje de aprendizaje automático. A continuación, se ofrece una visión general de los temas y cuadernos incluidos en este repositorio, destacando cómo cada uno aborda diferentes aspectos y funcionalidades de Langchain.

### Configuración inicial

Antes de sumergirnos en los cuadernos, es esencial establecer un archivo `config.py` en la carpeta correspondiente y colocar su clave API de OpenAI:

```python
||
```

## Cuadernos y temas clave

### Introducción a LangChain

#### **1. Preparación**

Esta sección se encarga de la instalación de `langchain` y `openai`, seguida de la configuración de la clave API de OpenAI en el entorno.

```python
%pip install langchain
%pip install openai
```

```python
import config
os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY
```

#### **2. Introducción a LangChain**

Se exponen los conceptos básicos de LangChain, incluyendo Model, Prompt Templates y Chain, aunque en esta sección no se incluye código específico.

#### **3. Modelos**

##### **3.1 LLMs (Language Learning Models)**

Muestra cómo utilizar los modelos de OpenAI en LangChain, desde la inicialización hasta la generación de texto y la ejecución de ejemplos específicos.

```python
from langchain.llms import OpenAI
llm = OpenAI()
# Ejemplo de uso: llm.predict('Cuéntame un chiste.')
```

##### **3.2 ChatModels**

Explica cómo los ChatModels procesan una lista de mensajes, detallando los roles de cada entidad en la comunicación. El código muestra la creación de mensajes y la generación de plantillas para la interacción.

```python
from langchain.schema.messages import ChatMessage
# Ejemplo de creación de chat history y generación de una respuesta.
```

#### **4. Chain**

Demuestra cómo crear y utilizar una cadena en LangChain, incluyendo la creación de una plantilla de prompt y la ejecución de una cadena para generar nombres de empresas basados en una descripción.

```python
from langchain import PromptTemplate
from langchain.chains import LLMChain
# Ejemplo de creación y ejecución de una cadena.
```

### Langchain 1 - Modelos y Prompts

#### **1. Modelos**

Esta sección demuestra cómo cargar e interactuar con modelos de lenguaje de OpenAI utilizando LangChain, enfocándose en cómo se puede solicitar a estos modelos que generen texto.

```python
from langchain.llms import LlamaCpp, OpenAI
llm_openai = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=api)
respuesta_openai = llm_openai("Hola, como estas?")
print(respuesta_openai)
```

#### **2. Modelos Chat**

##### **2.1 ChatGTP**

Se introduce el uso de modelos específicos para chat, como ChatGPT, explicando cómo se pueden usar para generar respuestas más contextualizadas en diálogos.

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
chatgpt = ChatOpenAI(openai_api_key=api)
respuesta = chatgpt([HumanMessage(content="Hola, como estas?")])
print(respuesta.content)
```

#### **3. Prompts**

Explica la importancia de estructurar bien los prompts y cómo LangChain ofrece herramientas para facilitar esto, permitiendo la creación de templates y la incorporación de ejemplos para mejorar la precisión de las respuestas.

```python
from langchain import PromptTemplate
template_basico = """Eres un asistente virtual culinario...
prompt_temp = PromptTemplate(input_variables=["platillo"], template = template_basico)
```

##### **3.1 Ejemplo cuando no usamos PROMP de ejemplo**

Muestra la diferencia en la respuesta del modelo al no utilizar un prompt de ejemplo, destacando la utilidad de estos en proporcionar contexto al modelo.

```python
llm_openai("¿Cuál es el ingrediente principal de las quesadillas?")
```

#### 4. Output parser

LangChain permite parsear o formatear las respuestas del modelo para que sean más útiles o adecuadas para el contexto deseado.

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
output_parser = CommaSeparatedListOutputParser()
template_basico_parser = "Cuales son los ingredientes para preparar {platillo}\n{como_parsear}"
```

### Langchain 2 - Memoria

#### **1. Memoria**

Aqui se introduce el concepto de memoria en los LLMs, explicando cómo ayuda a los modelos a recordar información a largo plazo y describiendo diferentes tipos de memoria que se pueden implementar.

#### **2. Conversation buffer**

Esta memoria básica almacena cada interacción con el modelo. El historial completo se envía con cada nuevo prompt para ayudar al modelo a recordar interacciones previas.

```python
from langchain.memory import ConversationBufferMemory
memoria = ConversationBufferMemory()
chatbot = ConversationChain(llm=llm, memory=memoria, verbose=True)
chatbot.predict(input="Hola como estas?, Me llamo Marcos y soy un programador")
```

#### **3. Conversation buffer window memory**

Similar a la memoria del buffer de conversación, pero limita el historial a una ventana específica de mensajes, evitando sobrecargar el modelo con información innecesaria.

```python
from langchain.memory import ConversationBufferWindowMemory
memoria = ConversationBufferWindowMemory(window_size=5)  # Ejemplo de tamaño de ventana
```

#### **4. Conversation summary memory**

En lugar de enviar todo el historial, esta memoria envía un resumen de la conversación, permitiendo que el modelo recuerde el contexto sin sobrepasar su límite de tokens.

```python
from langchain.memory import ConversationSummaryMemory
memoria = ConversationSummaryMemory(llm=llm)
chatbot_resumen = ConversationChain(llm=llm, memory=memoria, verbose=True)
```

##### **4.1 Resumen de la conversación**

Demuestra cómo la memoria de resumen de conversación solo envía un resumen en lugar del historial completo al modelo.

```python
memoria.chat_memory.messages  # Muestra los mensajes en memoria
```

#### **5. Conversation Knowledge Graph Memory**

Implementa un grafo de conocimiento, almacenando piezas clave de la conversación para que el modelo pueda referenciar y responder con base en ese contexto.

```python
from langchain.memory import ConversationKGMemory
memoria = ConversationKGMemory(llm=llm)
chatbot_kgm = ConversationChain(llm=llm, memory=memoria, verbose=True)
```

##### **5.1 Detalles del grafo de conocimiento**

Muestra cómo se almacena la información clave en un grafo de conocimiento.

```python
print(chatbot_kgm.memory.kg.get_triples())  # Muestra los triples almacenados en el grafo
```

### Langchain 3 - Cadenas

#### **1. Cadenas**

Esta sección introduce el concepto de cadenas en LangChain, que permite crear flujos de trabajo combinando distintos "bloques" para crear sistemas con LLMs más complejos. Se menciona cómo las cadenas permiten gestionar qué modelo genera qué información, cómo se utilizan los prompts y cómo la salida de un modelo puede funcionar como entrada para otro.

#### **2. Cadenas más usadas**

Se presentan las cadenas más comunes y útiles integradas en LangChain, que facilitan el desarrollo de diversos sistemas.

##### **2.1 LLMChain**

Describe cómo LLMChain facilita la interacción con LLMs combinando un modelo y los templates de prompts. Se muestra cómo se puede utilizar LLMChain para generar respuestas en base a un tema específico.

```python
from langchain import LLMChain, OpenAI, PromptTemplate
llm = OpenAI(openai_api_key=API)
cadena_LLM = LLMChain(llm=llm, prompt=template)
cadena_LLM.predict(tema="ingenieria civil")
```

##### **2.2 SequentialChain**

Explica cómo SequentialChain permite crear secuencias de operaciones donde la salida de una cadena se convierte en la entrada de la siguiente, proporcionando un ejemplo de cómo se pueden encadenar dos procesos para obtener una recomendación de aprendizaje.

```python
from langchain.chains import SequentialChain
cadenas = SequentialChain(chains=[cadena_lista, cadena_inicio])
cadenas({"tema": "programacion"})
```

#### **3. Otros ejemplos**

Se proporcionan ejemplos adicionales de tipos de cadenas en LangChain, como MathChain y TransformChain, mostrando cómo se pueden realizar operaciones matemáticas o transformaciones de texto.

```python
# Ejemplo de MathChain
from langchain import LLMMathChain
cadena_mate = LLMMathChain(llm=llm, verbose=True)
cadena_mate.run("Cuanto es 432*12-32+32?")

# Ejemplo de TransformChain
from langchain.chains import TransformChain
cadena_transformacion = TransformChain(transform=eliminar_brincos)
cadena_transformacion.run(prompt)
```

## Fine tuning

Para hacerlo debes de instalar pandas primero y hacer tu API KEY de open AI en una variable de ambiente para que el CLI de openai pueda procesar tus peticiones

```code
pip install pandas
export OPENAI_API_KEY="<OPENAI_API_KEY>"
```

Una vez hecho esto, tienes que generar un archivo con tus prompts y el texto que debería aprender a responder GPT. Como en el ejemplo **./finetuning/haikus.csv**

### **Convertir .csv a jsonl**

Para hacer el fine tuning se requiere tener nuestros prompts y texto a completar en un formato jsonl.

Para esto podemos correr el siguiente comando

```code
openai tools fine_tunes.prepare_data -f <ARCHIVO>
```

### **Realizar el fine tuning**

Ahora que ya tenemos el archivo en formato jsonl lo unico que debemos hacer es correr este comando y esperar unos minutos.

```code
openai api fine_tunes.create -t <ARCHIVO_JSONL> -m <MODELO_SOBRE_EL_CUAL_HACER_FINE_TUNING>
```

Un vez que termina de entrenar nos va a desplegar en la terminal el nombre del nuevo modelo o lo podemos ver dentro de el playground de openAI.

Para poderlo usar en vez de usar el nombre de modelo ada (por ejemplo), se pone el nombre del modelo que ya fue ajustado.

## Embeddings

El proceso de embeddings transforma palabras o texto en vectores de N dimensiones, incorporando información semántica. Esto significa que palabras o textos semánticamente similares estarán cercanos en el espacio vectorial. El cuaderno utiliza específicamente el embedding de OpenAI, destacado por su capacidad para posicionar palabras o textos según su semántica, similar al utilizado por GPT-3.

### Secciones principales:

Claro, profundizaré en la descripción de cada sección del cuaderno "embeddings.ipynb", explicando el propósito y el contenido clave, incluido el código de Python cuando sea relevante.

#### **1. Instalaciones**

Esta sección se centra en la instalación de los paquetes necesarios para trabajar con embeddings y otras funcionalidades del cuaderno. Incluye comandos de instalación para las bibliotecas requeridas, asegurando que el entorno esté preparado para los ejemplos y aplicaciones siguientes.

```python
%pip install -U email-validator
%pip install gradio
%pip install pypdf
```

#### **2. Embeddings con OpenAI**

![embeddings](https://cdn.openai.com/new-and-improved-embedding-model/draft-20221214a/vectors-1.svg)

En esta parte, se introduce el concepto de embeddings y se explica cómo OpenAI facilita este proceso.

```python
from openai import Embedding
embedding = Embedding('text-embedding-model')
vector = embedding.encode("ejemplo de texto")
```

#### **3. Qué es y cómo usar embeddings**

Aquí se detalla qué significa hacer un embedding de datos y cómo se utilizan estos vectores numéricos en aplicaciones prácticas. Se explora la conversión de datos a vectores y cómo estos reflejan la similitud semántica.

```python
# Se puede hacer embeeding de palabras o cadenas de texto
palabras = ["casa", "perro", "gato", "lobo", "leon", "zebra", "tigre"]
```

```python
diccionario = {}
for i in palabras:
    diccionario[i] = get_embedding(i, engine="text-embedding-3-small")
```

```python
palabra = "gato"
print("Primeros 10 valores de {}:\n".format(palabra), diccionario[palabra][:10])
print("\n")
print("Número de dimensiones del dato embebido\n", len(diccionario[palabra]))
```

#### **4. Comparar dos embeddings**

Se muestra cómo calcular la distancia entre dos vectores, lo que indica qué tan cercanos o lejanos están en términos de su significado.

```python
# Código hipotético para comparar vectores
import numpy as np
similitud = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
```

```python
n_palabra = "gato" # Palabra nueva a comparar
palabra_comparar = "perro" # Palabra del diccionario con la que compararemos la nueva palabra
n_palabra_embed = get_embedding(n_palabra, engine="text-embedding-3-small")
similitud = cosine_similarity(diccionario[palabra_comparar], n_palabra_embed)
print(similitud)
```

#### **5. Sumar embeddings**

Se explica cómo la suma de vectores de embeddings puede crear nuevos vectores que representan conceptos combinados de los elementos sumados, proporcionando ejemplos de cómo se puede realizar y utilizar esta operación.

```python
# Suma dos listas usando pandas
sumados = (pd.DataFrame(diccionario["leon"])) + (pd.DataFrame(diccionario["zebra"]))
len(sumados)

for key, value in diccionario.items():
    print(key, ":", cosine_similarity(diccionario[key], sumados))
```

#### **6. Aplicación de un Chatbot**

En esta parte, se ilustra la aplicación práctica de embeddings en la construcción de un chatbot. Se utiliza Gradio para crear una interfaz donde los usuarios pueden interactuar con el chatbot, haciendo preguntas y recibiendo respuestas basadas en el análisis de embeddings.

```python
# Código hipotético para integrar Gradio y el chatbot
import gradio as gr
def chatbot_respuesta(pregunta):
    # Proceso para generar una respuesta
    return "Respuesta"
iface = gr.Interface(fn=chatbot_respuesta, inputs="text", outputs="text")
iface.launch()
```

#### **7. Procesar datos de un PDF**

Esta sección demuestra cómo los embeddings pueden aplicarse para extraer y analizar información de documentos PDF. Se muestra cómo leer un PDF, convertir su contenido en vectores de embeddings y utilizar estos para responder consultas basadas en el contenido del documento.

```python
# Código hipotético para procesar un PDF y realizar consultas
import PyPDF2
# Leer y procesar el PDF
# Convertir el contenido en embeddings
# Utilizar embeddings para responder consultas

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

loader = PyPDFLoader("./archivo_respuestas.pdf")
pages = loader.load_and_split()
```
