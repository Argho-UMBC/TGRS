import numpy as np
import json
import os
import argparse
import text_helper
from collections import defaultdict


def extract_answers(q_answers, valid_answer_set):
    all_answers = [str(q_answers)]
    valid_answers = [a for a in all_answers if a in valid_answer_set]
    return all_answers, valid_answers
# ,'FloodNet-vqa_test2022_risk_assessment','FloodNet-vqa_test2022_density_estimation','FloodNet-vqa_test2022_road-condition','FloodNet-vqa_test2022_building_condition','FloodNet-vqa_test2022_entire_image']:

def vqa_processing(image_dir, annotation_file, valid_answer_set, image_set):
    print('building vqa %s dataset' % image_set)
    if image_set in ['FloodNet-vqa_test2022_language_bias']:
        load_answer = True
        with open(annotation_file%image_set) as f:
            annotations = json.load(f)
            qid2ann_dict = {ann['Question_ID']: ann for ann in annotations}
    else:
        load_answer = False
#     print(qid2ann_dict)
    with open(annotation_file % image_set) as f:
        questions = json.load(f)
#     coco_set_name = image_set.replace('-dev', '')
    abs_image_dir = os.path.abspath(image_dir % 'FloodNet-vqa_test2022')
#     image_name_template = 'COCO_'+coco_set_name+'_%012d'
    dataset = [None]*len(questions)
    
    unk_ans_count = 0
    for n_q, q in enumerate(questions):
        if (n_q+1) % 5000 == 0:
            print('processing %d / %d' % (n_q+1, len(questions)))
        image_id = q['Image_ID']
        attention_dir="C:\\Users\\asarkar2\\argho\\Harvey_VQA\\Attention_Map\\"+ q['AttentionMap_dir']
        question_id = q['Question_ID']
        ques_type=q["Question_Type"]
#         image_name = image_name_template % image_id
        image_path = os.path.join(abs_image_dir, image_id)
        question_str = q['Question']
        question_tokens = text_helper.tokenize(question_str)
#         print(image_id)
#         print(image_path)
#         print(question_id)
#         print(question_str)
#         print(question_tokens)
        
        iminfo = dict(image_name=image_id,
                      image_path=image_path,
                      attention_path=attention_dir,
                      question_type=ques_type,
                      question_str=question_str,
                      question_tokens=question_tokens)
        
        if load_answer:
            ann = qid2ann_dict[question_id]
            all_answers, valid_answers = extract_answers(ann['Ground_Truth'], valid_answer_set)
#             print(all_answers)
#             print(valid_answers)
            
            if len(valid_answers) == 0:                                                                         
                valid_answers = ['<unk>']
                unk_ans_count += 1
            iminfo['all_answers'] = all_answers
            iminfo['valid_answers'] = valid_answers
            
        dataset[n_q] = iminfo
#         print(dataset)
    print('total %d out of %d answers are <unk>' % (unk_ans_count, len(questions)))
#         print(dataset)
    return dataset


def main(args):
    
    image_dir = 'C:\\Users\\asarkar2\\argho\\Harvey_VQA\\'+'Original_Image_Final_Reshape\\%s\\'
    annotation_file = args.input_dir+'questions\\%s_annotations.json'
#     question_file = args.input_dir+'/questions/%s_annotations.json'

    vocab_answer_file = args.output_dir+'vocab_answers.txt'
    answer_dict = text_helper.VocabDict(vocab_answer_file)
    valid_answer_set = set(answer_dict.word_list) 
    print(valid_answer_set)   
    
    # train = vqa_processing(image_dir, annotation_file, valid_answer_set, 'CARDS-vqa_training2022')
    # valid = vqa_processing(image_dir, annotation_file, valid_answer_set, 'CARDS-vqa_valid2022')
    # test = vqa_processing(image_dir, annotation_file, valid_answer_set, 'CARDS-vqa_test2022')
    test_lb=vqa_processing(image_dir, annotation_file, valid_answer_set, 'FloodNet-vqa_test2022_language_bias')
    # test_cc=vqa_processing(image_dir, annotation_file, valid_answer_set, 'FloodNet-vqa_test2022_complex_counting')
    # test_ra=vqa_processing(image_dir, annotation_file, valid_answer_set, 'FloodNet-vqa_test2022_risk_assessment')
    # test_de=vqa_processing(image_dir, annotation_file, valid_answer_set, 'FloodNet-vqa_test2022_density_estimation')
    # test_rc=vqa_processing(image_dir, annotation_file, valid_answer_set, 'FloodNet-vqa_test2022_road_condition')
    # test_bc=vqa_processing(image_dir, annotation_file, valid_answer_set, 'FloodNet-vqa_test2022_building_condition')
    # test_ei=vqa_processing(image_dir, annotation_file, valid_answer_set, 'FloodNet-vqa_test2022_entire_image')
    # test_bc=vqa_processing(image_dir, annotation_file, valid_answer_set, 'FloodNet-vqa_test2022_building_condition')
    
    
    
    
    
    
#     test_dev = vqa_processing(image_dir, annotation_file, question_file, valid_answer_set, 'test-dev2015')
    
    # np.save(args.output_dir+'/train.npy', np.array(train))
    # np.save(args.output_dir+'/valid.npy', np.array(valid))
    # np.save(args.output_dir+'/train_valid.npy', np.array(train+valid))
    # np.save(args.output_dir+'/test.npy', np.array(test))
    np.save(args.output_dir+'/test_language_bias.npy', np.array(test_lb))

    # np.save(args.output_dir+'/test_complex_count.npy', np.array(test_cc))
    # np.save(args.output_dir+'/test_risk_assessment.npy', np.array(test_ra))
    # np.save(args.output_dir+'/test_density_estimation.npy', np.array(test_de))
    # np.save(args.output_dir+'/test_road_condition.npy', np.array(test_rc))
    # np.save(args.output_dir+'/test_building_condition.npy', np.array(test_bc))
    # np.save(args.output_dir+'/test_entire_image.npy', np.array(test_ei))
    # np.save(args.output_dir+'/test_building_condition.npy', np.array(test_bc))
    
    
    
    
    
#     np.save(args.output_dir+'/test-dev.npy', np.array(test_dev))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='C:\\Users\\asarkar2\\Bina_Lab\\Harvey_VQA\\torch\\',
                        help='directory for inputs')

    parser.add_argument('--output_dir', type=str, default='C:\\Users\\asarkar2\\Bina_Lab\\Harvey_VQA\\torch\\',
                        help='directory for outputs')
    
    args = parser.parse_args()

    main(args)
