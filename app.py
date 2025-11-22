from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenetv2, MobileNet_V2_Weights
from PIL import Image
import io
import os
import copy 

# === 1. Definição da arquitetura (agora parametrizada) ===  
class CNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # remove classifier final
        for param in self.model.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1280, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.fc_out = nn.Linear(128, n_classes) 

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x

# === 2. Carregar os DOIS modelos ===

# Modelo 1: Verificador
CAMINHO_MODELO_VERIFICADOR = r"C:\Users\Roberto\Documents\programacao\python\upx_6\cnn_model_general.pt" 
classes_verificacao = ['folhas', 'nao_folhas'] 
modelo_verificador = CNN(n_classes=len(classes_verificacao))
if os.path.exists(CAMINHO_MODELO_VERIFICADOR):
    modelo_verificador.load_state_dict(torch.load(CAMINHO_MODELO_VERIFICADOR, map_location=torch.device('cpu')))
else:
    print(f"AVISO: Arquivo '{CAMINHO_MODELO_VERIFICADOR}' não encontrado. O verificador não funcionará.")
modelo_verificador.eval()


# --- Modelo 2: Diagnóstico ---
CAMINHO_MODELO_DIAGNOSTICO = r"C:\Users\Roberto\Documents\programacao\python\upx_6\cnn_model.pt"
classes_diagnostico = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
modelo_diagnostico = CNN(n_classes=len(classes_diagnostico)) 
if os.path.exists(CAMINHO_MODELO_DIAGNOSTICO):
    modelo_diagnostico.load_state_dict(torch.load(CAMINHO_MODELO_DIAGNOSTICO, map_location=torch.device('cpu')))
else:
     print(f"AVISO: Arquivo '{CAMINHO_MODELO_DIAGNOSTICO}' não encontrado. O diagnóstico não funcionará.")
modelo_diagnostico.eval()


# === 3. Transforms para cada modelo ===

verificacao_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

diagnostico_transforms_base = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

diagnostico_transforms_flip = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

diagnostico_tta_transforms = [diagnostico_transforms_base, diagnostico_transforms_flip]



tratamentos = {
    "Tomato___Bacterial_spot": "Remover partes afetadas e aplicar fungicida à base de cobre.",
    "Tomato___Early_blight": "Remover folhas afetadas, rotação de culturas, aplicar fungicida (clorotalonil).",
    "Tomato___Late_blight": "Remover plantas infectadas, aplicar fungicidas (mancozebe, clorotalonil).",
    "Tomato___Leaf_Mold": "Melhorar ventilação, reduzir umidade, aplicar fungicida (clorotalonil).",
    "Tomato___Septoria_leaf_spot": "Remover folhas infectadas, aplicar fungicida (clorotalonil).",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Aplicar acaricida e aumentar umidade.",
    "Tomato___Target_Spot": "Rotação de culturas e fungicidas protetores.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Controlar mosca-branca (vetor).",
    "Tomato___Tomato_mosaic_virus": "Evitar contato entre plantas, usar sementes certificadas.",
    "Tomato___healthy": "A planta está saudável, sem necessidade de tratamento."
}


app = FastAPI()

CONF_THRESHOLD = 0.7

@app.post("/classificar/")
async def classificar(file: UploadFile = File(...)):
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_para_diagnostico = image.copy() 
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"erro": f"Não foi possível ler a imagem: {str(e)}"}
        )

    try:
        img_tensor_verif = verificacao_transforms(image).unsqueeze(0)
        
        with torch.no_grad():
            saida_verif = modelo_verificador(img_tensor_verif)
            _, pred_verif = torch.max(saida_verif, 1)
            
        classe_verificada = classes_verificacao[pred_verif.item()]
        
        if classe_verificada == 'nao_folhas':
            return JSONResponse(
                status_code=400,
                content={
                    "erro": "Imagem inválida.",
                    "sugestao": "Por favor, envie uma foto de uma folha de tomate em uma superficie plana."
                }
            )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"erro": f"Erro durante a verificação da imagem: {str(e)}"}
        )

    try:
        all_probs = []
        with torch.no_grad():
            for t in diagnostico_tta_transforms:
                img_tensor_diag = t(image_para_diagnostico).unsqueeze(0) 
                saida_diag = modelo_diagnostico(img_tensor_diag)
                probs = F.softmax(saida_diag, dim=1)
                all_probs.append(probs)

        probs_medias = torch.mean(torch.cat(all_probs, dim=0), dim=0)
        conf, pred_diag = torch.max(probs_medias, 0)
        
        confianca_percentual = round(conf.item() * 100, 2)

        if conf.item() < CONF_THRESHOLD:
            return JSONResponse(
                status_code=400,
                content={
                    "erro": "Não foi possível identificar a doença com confiança.",
                    "sugestao": "Por favor, tente tirar uma nova foto com melhor iluminação e foco.",
                    "confianca": confianca_percentual
                }
            )
        
        doenca = classes_diagnostico[pred_diag.item()]
        tratamento = tratamentos.get(doenca, "Tratamento não encontrado.")

        return JSONResponse(content={
            "doenca": doenca,
            "tratamento": tratamento,
            "confianca": confianca_percentual
        })

    except Exception as e:
         return JSONResponse(
            status_code=500, 
            content={"erro": f"Erro durante o diagnóstico da doença: {str(e)}"}
        )
