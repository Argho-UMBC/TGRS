import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader import get_loader
from Attention_Models import VqaModel
import random
# def set_deterministic(seed=12):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic=True
#     torch.backends.cudnn.benchmark=False
#     torch.backends.cudnn.enabled=False
    
# set_deterministic(seed=12)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def embedding(glove_filename,vocab_file): 
#     # word_index=vocab_file # call my_token function

#     with open(vocab_file) as f:
#         lines = f.readlines()
#     lines = [l.strip() for l in lines]

#     embeddings_index = dict()
#     f = open(glove_filename,encoding='utf8') # call glove vector text file
#     for line in f:
#         values=line.split() # split the each line in text file
#         word = values[0] # first index associate with word and othe other indexs represent embedding vector associated with that word 
#         coefs = np.asarray(values[1:], dtype='float32') 
#         embeddings_index[word] = coefs
#     f.close()

#     vocab_size=len(lines)
#     embedding_matrix=np.zeros((vocab_size+1,300)) # define embedding matrix
#     # print(word_index.items())
#     for word,i in enumerate(lines):                # create our embedding matrix for each word of questions.
#         embedding_vector = embeddings_index.get(str(word))
# #         print(embedding_vector)
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector
#     return embedding_matrix

# token_data=QuestionPreprocess.tokenize_question(ques_train)

# emb_mat=embedding(root_dir+"\\glove.6B.300d.txt",vocab_file)

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/dim
    return torch.exp(-kernel_input) # (x_size, y_size)

#computes mmd loss
def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
        
def main(args):
    if not os.path.exists(args.log_dir.replace("/logs","")):
        os.mkdir(args.log_dir.replace("/logs",""))
        
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
        
#     if not os.path.exists(args.model_dir.replace("/models","")):
#         os.mkdir(add.model_dir.replace("/models",""))
        
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
        
            

    data_loader = get_loader(
        input_dir=args.input_dir,
        input_vqa_train='train.npy',
        input_vqa_valid='valid.npy',
        max_qst_length=args.max_qst_length,
        max_num_ans=args.max_num_ans,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
    ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size
    ans_unk_idx = data_loader['train'].dataset.ans_vocab.unk2idx

    model = VqaModel(
        embed_size=args.embed_size,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=args.word_embed_size,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
    model_name=args.model_name,
        pre_train=args.pre_train,
    spatial=args.spatial
    ).to(device)

    criterion1 = nn.CrossEntropyLoss()
    criterion2=nn.KLDivLoss(reduction="batchmean")

    # criterion2=nn.L1Loss()

    # criterion2=nn.MSELoss()
    
    
    params=list(model.parameters())

#     params = list(model.pre_img_encoder.parameters()) \
#             + list(model.post_img_encoder.parameters())\
#         + list(model.qst_encoder.parameters()) \
#         + list(model.fc1.parameters()) \
#         + list(model.fc2.parameters())

    optimizer = optim.Adam(params, lr=args.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in range(args.num_epochs):

        for phase in ['train', 'valid']:

            running_loss = 0.0
            running_corr_exp1 = 0
            running_corr_exp2 = 0
            batch_step_size = len(data_loader[phase].dataset) / args.batch_size

            if phase == 'train':

                scheduler.step()
                
                model.train()
            else:
                model.eval()

            for batch_idx, batch_sample in enumerate(data_loader[phase]):

                # print(batch_sample)

                img = batch_sample['image'].to(device)
                question = batch_sample['question'].to(device)
                label = batch_sample['answer_label'].type(torch.LongTensor).to(device)
# #                 label=label.view(-1,196)
                gt_atten=batch_sample['attention'].to(device)/batch_sample['attention'].squeeze().sum(1).sum(1).unsqueeze(1).unsqueeze(1).to(device)
                gt_atten=gt_atten.view(-1,196)
                                         
#                 label=label.type(torch.LongTensor)
#                 print(gt_atten)
#                 print(label.shape)
                # print(question.shape)
#                 print(img.shape)
                
                
                multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    output,sfmax = model(img, question)
                    sfmax=sfmax.view(-1,196)
#                     print(output.shape)
#                     print(sfmax.shape)
#                     .shape
#                     output=output.long()
                    # [batch_size, ans_vocab_size=1000]
                    _, pred_exp1 = torch.max(output, 1)  # [batch_size]
                    _, pred_exp2 = torch.max(output, 1)  # [batch_size]
                    loss1 = criterion1(output, label)
                    loss2=criterion2(sfmax.log(), gt_atten)
                    # loss2=criterion2(sfmax, gt_atten)
                    
                    final_loss=loss1+(args.lambda_weight*loss2)

                    if phase == 'train':
                        final_loss.backward()
                        clip_gradient(model, 1)
                        optimizer.step()

                # Evaluation metric of 'multiple choice'
                # Exp1: our model prediction to '<unk>' IS accepted as the answer.
                # Exp2: our model prediction to '<unk>' is NOT accepted as the answer.
                pred_exp2[pred_exp2 == ans_unk_idx] = -9999
                running_loss += final_loss.item()
                running_corr_exp1 += torch.stack([(ans == pred_exp1.cpu()) for ans in multi_choice]).any(dim=0).sum()
                running_corr_exp2 += torch.stack([(ans == pred_exp2.cpu()) for ans in multi_choice]).any(dim=0).sum()

                # Print the average loss in a mini-batch.
                if batch_idx % 100 == 0:
                    print('| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}'
                          .format(phase.upper(), epoch+1, args.num_epochs, batch_idx, int(batch_step_size), final_loss.item()))

            # Print the average loss and accuracy in an epoch.
            epoch_loss = running_loss / batch_step_size
            epoch_acc_exp1 = running_corr_exp1.double() / len(data_loader[phase].dataset)      # multiple choice
            epoch_acc_exp2 = running_corr_exp2.double() / len(data_loader[phase].dataset)      # multiple choice

            print('| {} SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Acc(Exp1): {:.4f}, Acc(Exp2): {:.4f} \n'
                  .format(phase.upper(), epoch+1, args.num_epochs, epoch_loss, epoch_acc_exp1, epoch_acc_exp2))

            # Log the loss and accuracy in an epoch.
            with open(os.path.join(args.log_dir, '{}-log-epoch-{:02}.txt')
                      .format(phase, epoch+1), 'w') as f:
                f.write(str(epoch+1) + '\t'
                        + str(epoch_loss) + '\t'
                        + str(epoch_acc_exp1.item()) + '\t'
                        + str(epoch_acc_exp2.item()))
#             with open(os.path.join(args.sfmax_dir, '{}-log-epoch-{:02}.txt')
#                       .format(phase, epoch+1), 'w') as f:
#                 f.write(str(epoch+1) + '\t'
#                         + str(epoch_loss) + '\t'
#                         + str(epoch_acc_exp1.item()) + '\t'
#                         + str(epoch_acc_exp2.item()))

        # Save the model check points.
        if (epoch+1) % args.save_step == 0:
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()},
                       os.path.join(args.model_dir, 'model-epoch-{:02d}.ckpt'.format(epoch+1)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default="C:\\Users\\asarkar2\\Bina_Lab\\Harvey_VQA\\torch\\",
                        help='input directory for visual question answering.')

    parser.add_argument('--log_dir', type=str, default="C:\\Users\\asarkar2\\Bina_Lab\\Harvey_VQA\\torch\\",
                        help='directory for logs.')

    parser.add_argument('--model_dir', type=str, default="C:\\Users\\asarkar2\\Bina_Lab\\Harvey_VQA\\torch\\",
                        help='directory for saved models.')
    
    parser.add_argument('--model_name', type=str, default="vgg16",
                        help='image_model.')

    parser.add_argument('--modal_fusion', type=str, default="mul",
                        help='modal fusion.')
    
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
    
    parser.add_argument('--lambda_weight', type=float, default=0,
                        help='weight of loss 2.')
    

    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers of the RNN(LSTM).')

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
    # param = [0,1,10,50,100,150,200]

    param=[100,200]
    
    # m_name=["vgg16","resnet50","resnet152"]
    m_name=["resnet50"]
    for j in m_name:
        for i in param:
            
            args.lambda_weight = i
            args.model_name=j
            print("running for model ", j)
            print("running for lambda weight ",i)
            args.log_dir="G:\\My Drive\\Back_Up Research\\TGRS\\Ablation\\{}\\wo_pre_train\\lstm3\\w_{}\\logs".format(j,i)
            args.model_dir="G:\\My Drive\\Back_Up Research\\TGRS\\Ablation\\{}\\wo_pre_train\\lstm3\\w_{}\\models".format(j,i)

            main(args)
