import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader_test import get_loader
from Attention_Models import VqaModel
import random


# m_name="resnet50" 
# lstm_layer=1
# p_train="wo_pre_train"
# m_name_cap="VGG-16"
# l_layer_cap="1-Layer LSTM"



root_dir="G:\\My Drive\\Back_UP Research\\TGRS\\Ablation\\"



# metric=[]
# w=[0,1,10,50,100,150,200]
# for z in w:
#     for j in ["train","valid"]:
#         for i in range(30):
#             if i<9:
#                 txt=open(root_dir+"{}\\{}\\lstm{}\\w_{}\\logs\\{}-log-epoch-0{}.txt".format(m_name,p_train,lstm_layer,z,j,i+1))
#                 val=float(txt.read().split("\t")[2])
#                 metric.append(val)
#             else:
#                 txt=open(root_dir+"{}\\{}\\lstm{}\\w_{}\\logs\\{}-log-epoch-{}.txt".format(m_name,p_train,lstm_layer,z,j,i+1))
#                 val=float(txt.read().split("\t")[2])
#                 metric.append(val)



# train_metric_w0=metric[:30]
# valid_metric_w0=metric[30:60]
# train_metric_w1=metric[60:90]
# valid_metric_w1=metric[90:120]
# train_metric_w10=metric[120:150]
# valid_metric_w10=metric[150:180]
# train_metric_w50=metric[180:210]
# valid_metric_w50=metric[210:240]
# train_metric_w100=metric[240:270]
# valid_metric_w100=metric[270:300]
# train_metric_w150=metric[300:330]
# valid_metric_w150=metric[330:360]
# train_metric_w200=metric[360:390]
# valid_metric_w200=metric[390:420]
# # train_metric_w250=metric[420:450]
# # valid_metric_w250=metric[450:480]

# 'test.npy','test_simple_count.npy','test_complex_count.npy','test_risk_assessment.npy','test_density_estimation.npy','test_road_condition.npy','test_building_condition.npy','test_entire_image.npy']:
     
# j=50
# e=np.argmax(locals()["valid_metric_w{}".format(j)])

# print(e)

# e=np.argmin(locals()["valid_metric_w{}".foramt(j)])

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Function to test the model 
    # def test(): 
        # 'test.npy','test_simple_count.npy','test_complex_count.npy','test_risk_assessment.npy','test_density_estimation.npy','test_road_condition.npy','test_building_condition.npy',
    # j=args.lambda_weight   
    e= args.epoch
    #     for j in range(30):          
    for i in ["test_language_bias.npy"]:
        data_loader = get_loader(
            input_dir="C:\\Users\\asarkar2\\Bina_Lab\\Harvey_VQA\\torch\\",
            input_vqa_test=i,
    #         input_vqa_valid='valid.npy',
            max_qst_length=14,
            batch_size=32,
            max_num_ans=49,
            num_workers=1)

        qst_vocab_size = data_loader['test'].dataset.qst_vocab.vocab_size
        # print(qst_vocab_size)
        ans_vocab_size = data_loader['test'].dataset.ans_vocab.vocab_size
        # print(ans_vocab_size)
        ans_unk_idx = data_loader['test'].dataset.ans_vocab.unk2idx
        # Load the model that we saved at the end of the training loop 
        model = VqaModel(
            embed_size=1024,
            qst_vocab_size=qst_vocab_size,
            ans_vocab_size=49,
            word_embed_size=300,
            num_layers=args.num_layers,
            hidden_size=1024,
            model_name=args.model_name,
        pre_train=False,
        spatial=True).to(device) 


        if e<9:
            path = "G:\\My Drive\\Back_UP Research\\TGRS\\Ablation\\"+"{}\\{}\\lstm{}\\w_{}\\models\\model-epoch-0{}.ckpt".format(args.model_name,args.p_train,args.num_layers,args.lambda_weight,args.epoch+1) 
        else: 
            path = "G:\\My Drive\\Back_UP Research\\TGRS\\Ablation\\"+"{}\\{}\\lstm{}\\w_{}\\models\\model-epoch-{}.ckpt".format(args.model_name,args.p_train,args.num_layers,args.lambda_weight,args.epoch+1)

        model.load_state_dict(torch.load(path)["state_dict"])  

        model.eval()

        running_accuracy = 0 
        total = 0 
        # mse=[]

        i_id,g_val,p_val,q=[],[],[],[]
    #         print(list(data_loader['test']))

        with torch.no_grad(): 
            for batch_idx, batch_sample in enumerate(data_loader['test']):



                # pre_image = batch_sample['pre_image'].to(device)
                img = batch_sample['image'].to(device)
                image_id=batch_sample['image_id']
                question_str=batch_sample['question_str']
                
    #             print(pre_image)
                question = batch_sample['question'].to(device)
                label = batch_sample['answer_label'].to(device)
                multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.
                # gt_atten=batch_sample['attention'].to(device)
                gt_atten=batch_sample['attention'].to(device)/batch_sample['attention'].squeeze().sum(1).sum(1).unsqueeze(1).unsqueeze(1).to(device)
                gt_atten=gt_atten.view(-1,196).to(torch.float32)
                output,sfmax = model(img, question)
                sfmax=sfmax.view(-1,196).to(torch.float32)
    #             return output
                output = output.to(torch.float32) 

                # i_id.append(batch_sample["image_name"])
                # g_val.append(label)
                # p=n[np.arange(len(output.long())), label.long()].to("cpu")
                # print(output.size(0))
                # mse.append((torch.sum(torch.sum(torch.square(sfmax -gt_atten ),axis=1))/32).to("cpu"))
                # print(mse)
                # print(p.numpy().tolist())
                # for j in p:
                #     prob.append(j)

                # print(prob)
                _, predicted = torch.max(output, 1)
                # p_val.append(predicted) 
                for k in label:
                    g_val.append(k.item())
                for m in predicted:
                    p_val.append(int(m.item())+1)
                for n in image_id:
                    i_id.append(n)
                for t in question_str:
                    q.append(t)
        df=pd.DataFrame([i_id,q,g_val,p_val],index=["Image_ID","Question","Ground_Value","Predicted_Value"]).T

        df.to_csv("C:\\Users\\asarkar2\\Bina_Lab\\Harvey_VQA\\torch\\df_lb.csv")
    #             total += output.size(0) 
    #             running_accuracy += (predicted == label).sum().item() 
    # #             print(label) 

    #         # print("mse ", torch.mean(torch.tensor(mse))) 

    #         print('Accuracy_{}'.format(i.replace(".npy","")), (100 * running_accuracy / total))


            # plt.figure(figsize=(8,6))
            # sns.histplot(np.array(prob), kde=True)
            # plt.savefig(root_dir+"result\\score_0.png",dpi=500,bbox_inches="tight")
            # print(prob)


    # # test()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default="C:\\Users\\asarkar2\\Bina_Lab\\Harvey_VQA\\torch\\",
                        help='input directory for visual question answering.')

    

    parser.add_argument('--model_dir', type=str, default="G:\\My Drive\\Back_UP Research\\TGRS\\Ablation\\",
                        help='directory for saved models.')
    
    parser.add_argument('--model_name', type=str, default="resnet152",
                        help='image_model.')

    parser.add_argument('--lambda_weight', type=float, default=10,
                            help='weight of loss 2.')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers of the RNN(LSTM).')

    parser.add_argument('--epoch', type=int, default=25,
                        help='number of layers of the RNN(LSTM).')
    
    parser.add_argument('--spatial', type=bool, default=True,
                        help='spatial feature or not from image encoder.')
    

    parser.add_argument('--max_qst_length', type=int, default=14,
                        help='maximum length of question. \
                              the length in the VQA dataset = 26.')

    parser.add_argument('--max_num_ans', type=int, default=49,
                        help='maximum number of answers.')

    parser.add_argument('--embed_size', type=int, default=1024,
                        help='embedding size of feature vector \
                              for both image and question.')

    parser.add_argument('--word_embed_size', type=int, default=300,
                        help='embedding size of word \
                              used for the input in the LSTM.')
    
    parser.add_argument('--pre_train', type=bool, default=False,
                        help='Pre-train imagenet weight.')
    parser.add_argument('--p_train', type=str, default="wo_pre_train",
                        help='Pre-train imagenet weight.')
    
    
    

    

    parser.add_argument('--hidden_size', type=int, default=1024,
                        help='hidden_size in the LSTM.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for training.')

    parser.add_argument('--step_size', type=int, default=5,
                        help='period of learning rate decay.')

    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multiplicative factor of learning rate decay.')

    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs.')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size.')

    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of processes working on cpu.')

    parser.add_argument('--save_step', type=int, default=1,
                        help='save step of model.')

    args = parser.parse_args()
    
    main(args)
