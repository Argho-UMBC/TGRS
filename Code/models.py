import torch
import torch.nn as nn
import torchvision.models as models
import fusion
import torch.nn.functional as F

def embedding(glove_filename,vocab_file): 
    # word_index=vocab_file # call my_token function

    with open(vocab_file) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]

    embeddings_index = dict()
    f = open(glove_filename,encoding='utf8') # call glove vector text file
    for line in f:
        values=line.split() # split the each line in text file
        word = values[0] # first index associate with word and othe other indexs represent embedding vector associated with that word 
        coefs = np.asarray(values[1:], dtype='float32') 
        embeddings_index[word] = coefs
    f.close()

    vocab_size=len(lines)
    embedding_matrix=np.zeros((vocab_size+1,300)) # define embedding matrix
    # print(word_index.items())
    for word,i in enumerate(lines):                # create our embedding matrix for each word of questions.
        embedding_vector = embeddings_index.get(str(word))
#         print(embedding_vector)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

weight=embedding("C:\\Users\\asarkar2\\argho\\Harvey_VQA\\glove.6B.300d.txt","C:\\Users\\asarkar2\\Bina_Lab\\Harvey_VQA\\torch\\vocab_questions.txt")

class ImgEncoder(nn.Module):

    def __init__(self, embed_size,model_name,pre_train=False,spatial=False):

        super(ImgEncoder, self).__init__()
        
        self.spatial=spatial
        self.model_name=model_name
        
        if self.model_name=="resnet101":
            model = models.resnet101(pretrained=pre_train)
            in_features = model.fc.in_features  # input size of feature vector
            model = nn.Sequential(*(list(model.children())[:-2]))    # remove last fc layer

        
        elif self.model_name=="resnet152":
            model = models.resnet152(pretrained=pre_train)
            in_features = model.fc.in_features  # input size of feature vector
            model = nn.Sequential(*(list(model.children())[:-2]))    # remove last fc layer

#             
        elif self.model_name=="vgg16":
            model = models.vgg16(pretrained=pre_train).features
            in_features = 512  # input size of feature vector
            model = nn.Sequential(*(list(model.children())))    # remove last fc layer

            
        elif self.model_name=="densenet121":
            model = models.densenet121(pretrained=pre_train).features
            in_features = 1024  # input size of feature vector
            pre_model = nn.Sequential(*(list(model.children())))    # remove last fc layer

        self.model = model 
        self.fc = nn.Linear(in_features,embed_size)    
        self.tanh=nn.Tanh()       

    def forward(self, image):
        """Extract feature vector from image vector.
        
        """
        
        img_feature = self.model(image)

        
        if self.spatial==False:
            feat_size=img_feature.size()
            feat_dim=img_feature.dim()
            if feat_dim==2:
                img_feature=self.fc(img_feature)
                
                img_feature=self.tanh(img_feature)


#                 return pre_img_feature
            else: 
                if feat_size[2]==1:
                    img_feature=torch.squeeze(img_feature,3)
                    img_feature=torch.squeeze(img_feature,2)
                    img_feature=self.fc(img_feature)
                    img_feature=self.tanh(img_feature)
                else: 
                    img_feature=torch.sum(img_feature,3)
                    img_feature=torch.sum(img_feature,2)
                    img_feature=self.fc(img_feature)
                    img_feature=self.tanh(img_feature)
                    
        else:
            if not img_feature.is_contiguous():
                img_feature = img_feature.contiguous()
            img_feature = img_feature.view(-1,img_feature.size()[1], img_feature.size()[2]*img_feature.size()[3]).transpose(1,2) # [batch_size, 196, 512]
            img_feature = self.fc(img_feature)
            img_feature=self.tanh(img_feature)

                          

        return img_feature

    
    



class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size,padding_idx=0)
        # self.word2vec=nn.Embedding.from_pretrained(weight,freeze=False)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question):

        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]
        qst_feature=self.tanh(qst_feature)
        return qst_feature
    
    
class MfbAttention(nn.Module):
    def __init__(self):
        super(MfbAttention, self).__init__()
        
        self.Linear_imgproj = nn.Linear(1024, 5000)
        self.Linear_qproj = nn.Linear(1024, 5000)
        self.fc=nn.Linear(1000,1024)
        self.tanh=nn.Tanh()
        self.ff_attention = nn.Linear(1000, 1)
    def forward(self, vi,vq):
        iv = self.Linear_imgproj(vi)
        iv=self.tanh(iv)
        b_size=iv.size()[0]
        h_w=iv.size()[1]
        d_size=iv.size()[2]
                   
        iq = self.Linear_qproj(vq) 
        iq=self.tanh(iq)
        iq = iq.unsqueeze(1)
#         iq=iq.expand(b_size,h_w,d_size)
        mfb = torch.mul(iv, iq)
#         mfb = F.dropout(mfb, self.opt.MFB_DROPOUT_RATIO, training=self.training)
        mfb = mfb.view(-1, 1, 1000, 5)
        mfb = torch.sum(mfb, 3)                     # sum pool
        mfb = torch.sqrt(F.relu(mfb)) - torch.sqrt(F.relu(-mfb))       # signed sqrt
        mfb = F.normalize(mfb)
        mfb=mfb.view(b_size,h_w,-1)
#         ha=self.fc(mfb)
#         ha=self.tanh(ha)
        
        ha = self.ff_attention(mfb)
        pi = torch.softmax(ha, dim=1)
        self.pi = pi
        vi_attended = (pi * vi).sum(dim=1)
        u = vi_attended + vq
        
        
        return pi,u  
    
class SpatialMultiModalAttention(nn.Module):
    def __init__(self, num_channels, embed_size, dropout=True):
        """Stacked attention Module
        """
        super(MultiModalAttention, self).__init__()
        self.ff_image = nn.Linear(embed_size, num_channels)
        self.ff_questions = nn.Linear(embed_size, num_channels)
        self.ff_attention = nn.Linear(num_channels, 1)

    def forward(self, vi, vq):
        """Extract feature vector from image vector.
        """
        hi = self.ff_image(vi) 
        
        hq = self.ff_questions(vq).unsqueeze(1)
#         hq = hq.view(b_size,1,d_size)
#         hq=hq.expand(b_size,h_w,d_size)
        ha = torch.tanh(torch.mul(hi,hq))
        ha = self.ff_attention(ha)
        pi = torch.softmax(ha, dim=1)
        self.pi = pi
        vi_attended = (pi * vi).sum(dim=1)
        u = vi_attended + vq
        return u


class VqaModel(nn.Module):

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size,model_name,modal_fusion,pre_train=False,spatial=False):

        super(VqaModel, self).__init__()
        
        self.spatial=spatial
        self.mf=modal_fusion
        self.img_encoder = ImgEncoder(embed_size,model_name,pre_train,spatial)
        
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2=nn.Linear(embed_size,embed_size)
#         self.fc3 = nn.LazyLinear(embed_size)
#         if (self.mf=="mutan") or (self.mf=="mutan2d") or (self.ff=="mutan") or (self.ff=="mutan2d") :
#             self.fc3 = nn.Linear(512,embed_size)
#         else: 
        self.fc3=nn.Linear(embed_size,embed_size)
            
        if self.mf=="mlb":
            self.modal_fusion=fusion.MLBFusion({"dim_v":1024,"dim_q":1024,"dim_h":1024,"activation_v":"tanh",
                                                "activation_q":"tanh"})
        elif self.mf=="mutan":
            
            self.modal_fusion=fusion.MutanFusion({"dim_h":1024,"dim_mm":1024,"activation_v":"tanh","activation_q":"tanh",
                                                    "activation_hv":"tanh","activation_hq":"tanh","R":5})
            
        elif self.mf=="mlb2d":
            self.modal_fusion=fusion.MLBFusion2d({"dim_v":1024,"dim_q":1024,"dim_h":1024,"activation_v":"tanh",
                                                  "activation_q":"tanh"})
            
        elif self.mf=="mutan2d":
            self.modal_fusion=fusion.MutanFusion2d({"dim_h":1024,"dim_mm":1024,"activation_v":"tanh","activation_q":"tanh",
                                                    "activation_hv":"tanh","activation_hq":"tanh","R":5})
            
#         else:
#             self.modal_fusion=modal_fusion

    def forward(self, img, qst):
        
        img_feature = self.img_encoder(img)
        b_size=img_feature.size()[0]
        h_w=img_feature.size()[1]
        d_size=img_feature.size()[2]
        
        if self.spatial==False:
      
            qst_feature = self.qst_encoder(qst) 
         
            if self.mf=="mul":

                combined_feature=torch.mul(img_feature, qst_feature)

            elif self.mf=="concat":

                combined_feature=torch.cat([img_feature, qst_feature],dim=1)
                
            elif self.mf=="sum":

                combined_feature=torch.add(img_feature, qst_feature)
                
            elif self.mf=="subtract":

                combined_feature=torch.subtract(img_feature, qst_feature)
                
            else: 
                combined_feature=self.modal_fusion(img_feature, qst_feature)
            

            combined_feature = self.fc3(combined_feature)           # [batch_size, ans_vocab_size=1000]
            combined_feature = self.tanh(combined_feature)
            combined_feature = self.fc1(combined_feature) 

        
        
        
        else:    
            qst_feature = self.qst_encoder(qst) 
            qst_feature = qst_feature.view(b_size,1,d_size)
            qst_feature=qst_feature.expand(b_size,h_w,d_size)
            
            
            if self.mf=="mul":

                combined_feature=torch.mul(img_feature, qst_feature)

            elif self.mf=="concat":

                combined_feature=torch.cat([img_feature, qst_feature],dim=2)
                
            elif self.mf=="sum":

                combined_feature=torch.add(img_feature, qst_feature)

            elif self.mf=="subtract":

                combined_feature=torch.subtract(img_feature, qst_feature)
                
            
                
            else: 
                combined_feature=self.modal_fusion(img_feature, qst_feature)
                
            
#             combined_feature=torch.sum(combined_feature,1)
            combined_feature=self.fc3(combined_feature)
            combined_feature = self.tanh(combined_feature)
            combined_feature=torch.sum(combined_feature,1)
            combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]

        return combined_feature
