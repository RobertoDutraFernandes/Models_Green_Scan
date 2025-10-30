# Sistema de Diagnóstico de Doenças em Folhas de Tomate (IA)

Este repositório contém o pipeline de backend e os modelos de deep learning para um sistema de identificação de doenças em folhas de tomate. O sistema utiliza uma arquitetura de dois estágios para garantir precisão e segurança, sendo capaz de rejeitar imagens inválidas e, em seguida, diagnosticar 10 condições diferentes (incluindo folhas saudáveis).

O foco deste projeto foi criar um modelo robusto que superasse os desafios de overfitting comuns em datasets de imagens "de laboratório", para que pudesse generalizar e funcionar com imagens do "mundo real" (fotos de utilizadores).



## O Problema

O objetivo é simples: um utilizador tira uma foto de uma folha de tomate com o seu telemóvel, e a aplicação deve diagnosticar a doença (se houver) e sugerir um tratamento.

No entanto, esta tarefa apresenta um desafio técnico significativo:



- **Imagens "de Laboratório" vs. "Mundo Real":** A maioria dos datasets públicos disponíveis (como o PlantVillage) é "limpa" — as imagens mostram uma folha isolada sobre um fundo neutro e consistente (ex: um fundo cinza).

- **Imagens do Utilizador:** As fotos do mundo real são "sujas" — elas têm fundos complexos (terra, madeira, outras folhas), iluminação imprevisível e ângulos estranhos.

## A Abordagem Inicial e o Fracasso

A primeira abordagem (v1) foi um pipeline de transfer learning padrão:



- **Modelo:** Uma MobileNetV2 com as camadas-base congeladas (Feature Extraction).

- **Dataset:** Um grande dataset "limpo" (fundo cinza).

- **Resultado:** O modelo atingiu >99% de acerto no set de validação (que também tinha o fundo cinza).

**O Fracasso:** Quando testado no backend com fotos reais (ex: uma folha saudável num fundo de madeira), o modelo falhou catastroficamente.



- Ele classificava folhas saudáveis como "Septoria_leaf_spot" com 92% de confiança.

- Ele rejeitava fotos válidas de folhas como "nao_folhas".

**O Diagnóstico:** O modelo não aprendeu a identificar doenças. Ele aprendeu um "atalho" (shortcut): "Se o fundo for cinza, é uma folha de tomate." Quando o fundo cinza desaparecia, o modelo entrava em pânico e falhava.



## A Arquitetura de Pipeline Atual (Dois Modelos)

Para criar um sistema seguro e robusto, o backend implementa um pipeline de dois estágios:



**Modelo 1:** O Verificador (cnn_model_mixup_general.pt)

- **Propósito:** Um classificador binário (folhas vs. nao_folhas) que atua como um "segurança" (gatekeeper).

- **Função:** Rejeitar imagens que claramente não são folhas de tomate (ex: fotos de pessoas, animais, ou, como nos testes) antes que elas cheguem ao modelo de diagnóstico.

**Modelo 2:** O Diagnóstico (cnn_model_MIXUP.pt)

- **Propósito:** O modelo principal. Um classificador de 10 classes para as doenças (Bacterial_spot, Late_blight, healthy, etc.).

- **Função:** Recebe apenas as imagens que o Modelo 1 aprovou. Ele então analisa a folha e retorna a doença diagnosticada.

- **Camada de Segurança:** O backend só aceita um diagnóstico se a confiança do modelo for superior a um threshold (limite) estrito (ex: CONF_THRESHOLD = 0.80). Se a confiança for baixa (ex: 60%), o sistema informa ao utilizador que não pode identificar com clareza, em vez de dar um diagnóstico errado.

## A Estratégia de Treino SOTA (Estado-da-Arte)

Para resolver o problema do overfitting ao "fundo cinza", os dois modelos foram retreinados do zero usando um pipeline SOTA (Estado-da-Arte), focado em destruir "atalhos" e forçar a generalização.

Isto foi feito através de três técnicas principais no notebook de treino:



**Fine-Tuning (Ajuste Fino) da Arquitetura:**

Em vez de congelar toda a MobileNetV2, as últimas 20 camadas de features foram "descongeladas" (requires_grad = True). Isto permitiu ao modelo adaptar os seus "olhos" (treinados em cães e carros) para se especializarem em "texturas de fungos" e "formas de folhas".

**Data Augmentation Agressivo (Online):**

Foram aplicadas transformações aleatórias pesadas a cada imagem durante o treino:

- GaussianBlur: Simula fotos desfocadas.

- ColorJitter: Simula iluminação de telemóvel (brilho, contraste).

- RandomRotation: Simula fotos tortas.

- RandomErasing: A "primeira arma". Apaga aleatoriamente um "buraco" da imagem. Isto força o modelo a aprender a classificar, mesmo que partes da folha (ou do fundo) estejam em falta.

**Treino com Mixup (A Solução Definitiva):**

Esta foi a estratégia-chave. O Mixup mistura duas imagens (ex: 70% Imagem A + 30% Imagem B) e os seus rótulos (70% Rótulo A + 30% Rótulo B).

- **O Efeito**: Como o fundo cinza era uma constante em 100% das imagens (70% Cinza + 30% Cinza = 100% Cinza), o modelo aprendeu que o fundo era matematicamente inútil para resolver a equação da mistura de rótulos.

- **O Resultado**: O Mixup forçou o modelo a ignorar completamente o fundo e a focar-se apenas nas texturas das folhas "fantasmas" misturadas.

## O Resultado da Nova Estratégia

O novo modelo treinado (como visto nos testes do backend_test.py) é seguro e cauteloso.



- Ele já não é "confiante e errado".

- Quando vê uma imagem do mundo real que não reconhece, ele (corretamente) dá uma confiança baixa (ex: 40-70%).

- O nosso backend (com CONF_THRESHOLD = 0.80) filtra estas previsões incertas e pede ao utilizador uma foto melhor.

- Ele classifica corretamente as imagens do mundo real que consegue identificar com alta confiança.

## Limitações Atuais e Próximos Passos

Esta implementação representa a primeira instância (v1.0) de um modelo robusto, treinado com as melhores técnicas de generalização.

A principal limitação atual é a **escassez de dados de treino do "mundo real"**. Os datasets disponíveis na internet são limitados e sofrem dos mesmos "vícios" (bias) do fundo de laboratório.

O plano de ação para a v2.0 é usar esta aplicação como uma ferramenta de recolha de dados:



Permitir que os utilizadores enviem as suas fotos (as do mundo real).

Construir um "banco de dados" interno com estas novas imagens.

Periodicamente, rotular manualmente estas novas imagens e usá-las para retreinar e aprimorar os modelos.

À medida que o banco de dados do mundo real crescer, a confiança e a precisão dos modelos em imagens difíceis irão aumentar drasticamente.



transforme esses texte em um para readme utilizando markdown e o que for necessario