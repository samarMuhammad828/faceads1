import torch
from torchvision import transforms
from torch.autograd import Variable

class predictor:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
            ])
        self.modelname = ''
        
    def acc(self, a0,a1):
        x = round(((a0 - a1)/ (2 * a0)) * 100)
        x = max(x,0)
        x = min(x, 100)
        return str(x)+' %'
    

    def load_model(self,modelname):
        if modelname == 'gender':
            self.model =  torch.load('model_gender_ft.pkl', map_location='cpu')
            self.model.eval()
            self.modelname = 'gender'
        elif modelname == 'chubby':
            self.model = torch.load('modelChubby_ft.pkl', map_location='cpu')
            self.model.eval()
            self.modelname = 'chubby'
        elif modelname == 'glass':
            self.model = torch.load('modelglass_ft.pkl', map_location='cpu')
            self.model.eval()   
            self.modelname = 'glass'
        elif modelname == 'Receding_Hairline':
            self.model = torch.load('modelReceding_Hairline_ft.pkl', map_location='cpu')
            self.model.eval()          
            self.modelname = 'Receding_Hairline'
        elif modelname =='Bags_Under_Eyes':
            self.model = torch.load('modelBags_Under_Eyes_ft.pkl', map_location='cpu')
            self.model.eval()
            self.modelname ='Bags_Under_Eyes'
        elif modelname == 'Bald':
            self.model = torch.load('modelBald_ft.pkl', map_location='cpu')
            self.model.eval()
            self.modelname = 'Bald'
        elif modelname == 'Young':
            self.model = torch.load('modelYoung_ft.pkl', map_location='cpu')
            self.model.eval()
            self.modelname = 'Young'
        elif modelname == 'Pale_Skin':
            self.model = torch.load('modelPale_Skin_ft.pkl', map_location='cpu')
            self.model.eval()
            self.modelname = 'Pale_Skin'
            
        print(self.modelname +' is the model')
        
        
    def predict_gender(self,image):
        if self.modelname != 'gender':
            self.load_model('gender')
        image_tensor = self.test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        output = self.model(input)
        _, preds = torch.max(output, 1)

        index = output.data.cpu().numpy()
        a0 = index[0][0]
        a1 = index[0][1]
        if index.argmax() == 0:
            return ("woman", self.acc(a0,a1))
        else:
            return ("man",self.acc(a1,a0))
     

    def predict_look(self,image):
        if self.modelname != 'gender':
            self.load_model('gender')
        image_tensor = self.test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        output = self.model(input)
        _, preds = torch.max(output, 1)
        index = output.data.cpu().numpy()
        a0 = index[0][0]
        a1 = index[0][1]
        if index.argmax() == 0:
            return  ("You look like a famous woman with this accuracy", str(round(a0 * 30))+' %')
        else:
            return ("You look like a famous man with this accuracy", str(round(a1 * 30))+' %')
            
        
    
    
    def predict_glass(self,image):
        if self.modelname != 'glass':
            self.load_model('glass')
        image_tensor = self.test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        output = self.model(input)
        _, preds = torch.max(output, 1)
        index = output.data.cpu().numpy()
        a0 = index[0][0]
        a1 = index[0][1]
        if index.argmax() == 0:
            return ("wearing glasses", self.acc(a0,a1))
        else:
            return ("no glasses",self.acc(a1,a0))
        
    def predict_chubby(self,image):
        if self.modelname != 'chubby':
            self.load_model('chubby')
        image_tensor = self.test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        output = self.model(input)
        _, preds = torch.max(output, 1)
        index = output.data.cpu().numpy()
           
        a0 = index[0][0]
        a1 = index[0][1]
        
        if index.argmax() == 0:
            return ("Chubby", self.acc(a0,a1))
        else:
            return ("no Chubby",self.acc(a1,a0))
        
            
    def predict_Receding_Hairline(self,image):
        if self.modelname != 'Receding_Hairline':
            self.load_model('Receding_Hairline')
        image_tensor = self.test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        output = self.model(input)
        _, preds = torch.max(output, 1)

        index = output.data.cpu().numpy()
           
        a0 = index[0][0]
        a1 = index[0][1]
        
        if index.argmax() == 0:
            return ("no Receding_Hairline", self.acc(a0,a1))
        else:
            return ("Receding_Hairline",self.acc(a1,a0))    
        
    
    def predict_Bags_Under_Eyes(self,image):
        if self.modelname != 'Bags_Under_Eyes':
            self.load_model('Bags_Under_Eyes')
        image_tensor = self.test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        output = self.model(input)
        _, preds = torch.max(output, 1)

        index = output.data.cpu().numpy()
           
        a0 = index[0][0]
        a1 = index[0][1]
        
        if index.argmax() == 0:
            return ("Bags_Under_Eyes", self.acc(a0,a1))
        else:
            return ("no Bags_Under_Eyes",self.acc(a1,a0))
        
    def predict_Bald(self,image):
        if self.modelname != 'Bald':
            self.load_model('Bald')
        image_tensor = self.test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        output = self.model(input)
        _, preds = torch.max(output, 1)

        index = output.data.cpu().numpy()
           
        a0 = index[0][0]
        a1 = index[0][1]
        
        if index.argmax() == 0:
            return ("Bald", self.acc(a0,a1))
        else:
            return ("no Bald",self.acc(a1,a0))
        
    def predict_Young(self,image):
        if self.modelname != 'Young':
            self.load_model('Young')
        image_tensor = self.test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        output = self.model(input)
        _, preds = torch.max(output, 1)

        index = output.data.cpu().numpy()
           
        a0 = index[0][0]
        a1 = index[0][1]
        
        if index.argmax() == 0:
            return ("no Young", self.acc(a0,a1))
        else:
            return ("Young",self.acc(a1,a0))
        
    def predict_Pale_Skin(self,image):
        if self.modelname != 'Pale_Skin':
            self.load_model('Pale_Skin')
        image_tensor = self.test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        output = self.model(input)
        _, preds = torch.max(output, 1)

        index = output.data.cpu().numpy()
           
        a0 = index[0][0]
        a1 = index[0][1]
        
        if index.argmax() == 0:
            return ("no Pale_Skin", self.acc(a0,a1))
        else:
            return ("Pale_Skin",self.acc(a1,a0))
        