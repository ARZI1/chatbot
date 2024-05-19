# Conversational Chatbot Based On The Transformer Architecture


## About the project

This github repository contains the sources and documents for a final project in an artificial intelligence class. I decided to create a conversational chatbot based on the Generative Pre-trained Transformer (GPT) architecture underpinning most modern NLP models. The project is the culmination of over half a year’s worth of research and development. Documentation throughout the code was written in English, however the project paper was written in Hebrew.

In order to gain a better understanding of Natural Language Processing (NLP), and specifically the GPT architecture, I decided to implement most of the algorithms, mechanisms and layers from scratch. Research papers and blog posts were used as reference and cited in the project paper’s bibliography.

The tokenizer was implemented from scratch based on the Byte Pair Encoding (BPE) algorithm. In order to reduce tokenization time, a custom algorithm for encoding was created. In some cases it leads to inferior results, however the original implementation would have taken far too long. 

The different components of the Transformer unit were implemented from zero. I made a point out of writing the math behind the attention mechanisms to gain an intuition for how it works. Explanations about the different equations were explained extensively in the project paper. Furthermore, the positional encoding algorithms were also created from scratch with an emphasis on the math behind it.

Two models were trained in this project. During the training phase, I realized it would be impossible to train the model from scratch with the hardware at my disposal. To make matters worse, training the model using cloud computing solutions would be too costly, in the high hundreds. Therefore, I used a model with the same architecture as the one I created, but 12.5 times bigger and with some training already done. Both models are accessible in the application.

The first model was trained to write news articles based on the input a user gives it. The input can be the first word of the article, first line, first paragraph or be left blank. Dataset wise, the model was trained on some 312,000 news articles sourced from CNN. It boasts 6 transformer decoder units stacked atop each other, as well as embedding, encoding and dense layers. The model has a total of 10 million trainable parameters, and was trained on an RTX 2060 for around 40 hours.

The second model is an upscaled version of the first model, containing 12 transformer blocks and higher dimensionality for a grand total of 125 million trainable parameters. Because of its size, training such a model from scratch is infeasible. Therefore, a pretrained model was obtained from the Open Pretrained Transformers project, which was trained on a general dataset. In order to fine tune the model for question answering, a dataset was created by combining Stanford's Alpaca dataset with custom generated questions, for a total of 54,000 question-answer pairs. The model was trained for 8 hours on a T4 GPU provided by Google Colab.

The application is divided into three main parts. The first is the web server, powered by the Flask library and web sockets. The second is a distributed computing system. To enable multiple users to use the different models simultaneously a network of computers was constructed, each running a model instance. A central server keeps track of active computers and dispatches requests. The last part is used to connect the web server and computing system. It keeps track of active requests, matches requests to computers, sorts them by priority and does load balancing.

Because of their size, model weights are not included in this repository. That being said, the Google Colab notebooks linked in the demos section automatically download the weights in order to run the models.


## File structure

```
Project root
├── compute_server
│   ├── client
│   │   ├── ArticleModel.py
│   │   ├── ChatbotModel.py
│   │   ├── ComputeClient.py
│   │   ├── PositionalEmbedding.py
│   │   ├── Tokenizer.py
│   │   └── Transformer.py
│   ├── server
│   │   ├── ComputeMachine.py
│   │   └── ComputeServermanager.py
│   └── packets.py
├── request_manager
│   ├── Requests.py
│   └── RequestsManager.py
├── web_server
│   ├── WebServer.py
│   ├── static
│   │   ├── article_generator.css
│   │   ├── article_generator.js
│   │   ├── base.css
│   │   ├── chatbot.css
│   │   ├── chatbot.js
│   │   ├── compute_server.css
│   │   └── index.css
│   └── templates
│       ├── article_generator.html
│       ├── base.html
│       ├── chatbot.html
│       ├── compute_server.html
│       ├── index.html
│       └── login.html
└── main.py
```


## Demos

[Video demo](https://youtu.be/bFnjXwabQg4)

The following are Google Colab notebooks which contain code to load and use the models. Run all code cells, and scroll to the bottom. The final cell allows for model inference.

[Article Generator](https://colab.research.google.com/drive/1u18clYWDvUwG73hnYY7zmpeXeswVIOzZ?usp=sharing)

[Chatbot](https://colab.research.google.com/drive/1ISMZXIMaVswlp9vO2mV10xv-0fWXQuNg?usp=sharing)
