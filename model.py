import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

class DermNetModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = [
            'Acne and Rosacea Photos',
            'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
            'Atopic Dermatitis Photos',
            'Bullous Disease Photos',
            'Cellulitis Impetigo and other Bacterial Infections',
            'Eczema Photos',
            'Exanthems and Drug Eruptions',
            'Hair Loss Photos Alopecia and other Hair Diseases',
            'Herpes HPV and other STDs Photos',
            'Light Diseases and Disorders of Pigmentation',
            'Lupus and other Connective Tissue diseases',
            'Melanoma Skin Cancer Nevi and Moles',
            'Nail Fungus and other Nail Disease',
            'Poison Ivy Photos and other Contact Dermatitis',
            'Psoriasis pictures Lichen Planus and related diseases',
            'Scabies Lyme Disease and other Infestations and Bites',
            'Seborrheic Keratoses and other Benign Tumors',
            'Systemic Disease',
            'Tinea Ringworm Candidiasis and other Fungal Infections',
            'Urticaria Hives',
            'Vascular Tumors',
            'Vasculitis Photos',
            'Warts Molluscum and other Viral Infections'
        ]
        
        # Model yükleme - deprecated 'pretrained' parametresi yerine 'weights' kullan
        self.model = models.resnet50(weights=None)  # weights=None, pretrained=False ile aynı
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(self.classes))
        
        # Model ağırlıklarını yükle
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except Exception as e:
            raise Exception(f"Model dosyası yüklenemedi: {str(e)}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Görüntü ön işleme
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image):
        """
        Görüntüyü tahmin eder ve en yüksek olasılıklı sınıfı döndürür
        """
        # PIL Image'ı RGB'ye çevir (RGBA veya grayscale olabilir)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Transform ve batch dimension ekle
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
        return {
            'class': self.classes[predicted_class],
            'confidence': confidence
        } 