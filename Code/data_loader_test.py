import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from utils import text_helper
from skimage.color import rgb2gray

class VqaDataset(data.Dataset):

    def __init__(self, input_dir, input_vqa, max_qst_length=15, max_num_ans=33, transform=None,transform1=None):
        self.input_dir = input_dir
        self.vqa = np.load(input_dir+input_vqa,allow_pickle=True)
        self.qst_vocab = text_helper.VocabDict(input_dir+'vocab_questions.txt')
        self.ans_vocab = text_helper.VocabDict(input_dir+'vocab_answers.txt')
        self.max_qst_length = max_qst_length
        self.max_num_ans = max_num_ans
        self.load_ans = ('valid_answers' in self.vqa[0]) and (self.vqa[0]['valid_answers'] is not None)
        self.transform = transform
        self.transform1=transform1
#         self.post_transform = post_transform
        

    def __getitem__(self, idx):

        vqa = self.vqa
        qst_vocab = self.qst_vocab
        ans_vocab = self.ans_vocab
        max_qst_length = self.max_qst_length
        max_num_ans = self.max_num_ans
        transform = self.transform
        transform1 = self.transform1
        
        load_ans = self.load_ans

#         pre_image = vqa[idx]['image_path'].replace("post","pre")
#         pre_image = Image.open(pre_image).convert('RGB')
        attention_path=vqa[idx]['attention_path']
        attention=Image.open(attention_path).convert('RGB')
#         attention=rgb2gray(attention/255).reshape([-1])
        post_image = vqa[idx]['image_path']
        post_image = Image.open(post_image).convert('RGB')
        image_id=vqa[idx]["image_name"]
        question=vqa[idx]["question_str"]
        qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length)  # padded with '<pad>' in 'ans_vocab'
        qst2idc[:len(vqa[idx]['question_tokens'])] = [qst_vocab.word2idx(w) for w in vqa[idx]['question_tokens']]
        sample = {'image': post_image,'question': qst2idc,'attention':attention,'image_id':image_id,'question_str':question}

        if load_ans:
            ans2idc = [ans_vocab.word2idx(w) for w in vqa[idx]['valid_answers']]
            ans2idx = np.random.choice(ans2idc)
            sample['answer_label'] = ans2idx         # for training

            mul2idc = list([-1] * max_num_ans)       # padded with -1 (no meaning) not used in 'ans_vocab'
            mul2idc[:len(ans2idc)] = ans2idc         # our model should not predict -1
            sample['answer_multi_choice'] = mul2idc  # for evaluation metric of 'multiple choice'

        if transform:
#             sample['pre_image'] = transform(sample['pre_image'])
#         if post_transform:
            sample['image'] = transform(sample['image'])
        if transform1:
            sample['attention']=transform1(sample['attention']).squeeze()
            

        return sample

    def __len__(self):

        return len(self.vqa)


def get_loader(input_dir, input_vqa_test, max_qst_length, max_num_ans, batch_size, num_workers):

#     transform = {
#         phase: transforms.Compose([transforms.ToTensor(),
#                                    transforms.Normalize((0.485, 0.456, 0.406),
#                                                         (0.229, 0.224, 0.225))]) 
#         for phase in ['train', 'valid', 'test']}

    transform = {
        phase: transforms.Compose([transforms.ToTensor()
                               ]) 
        for phase in ['test']}
    
    transform1 = {
        phase: transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]) 
        for phase in ['test']}

    vqa_dataset = {
        'test': VqaDataset(
            input_dir=input_dir,
            input_vqa=input_vqa_test,
            max_qst_length=max_qst_length,
            max_num_ans=max_num_ans,
            transform=transform['test'],
        transform1=transform1['test'])
        }

    data_loader = {
        phase: torch.utils.data.DataLoader(
            dataset=vqa_dataset[phase],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)
        for phase in ['test']}

    return data_loader

