from attrdict import AttrDict
import glob, torch
from transformers import DistilBertTokenizerFast, AlbertTokenizerFast, BertTokenizerFast, RobertaTokenizerFast
from transformers import DistilBertForSequenceClassification, AlbertForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification
from bert_cls import read_data, read_data_split, label_fomatting, EMPDataset, evaluate
import argparse

def inference(args, infer_data, infer_label, mode='test'):

    print(args.output_dir)
    model_path_list = glob.glob(args.output_dir+'/checkpoint*')

    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)

    for i in range(len(model_path_list)):

        model = RobertaForSequenceClassification.from_pretrained(model_path_list[i])
        if torch.cuda.is_available():
            model.to(args.device)

        infer_encodings = tokenizer(infer_data, truncation=True, padding=True)
        label2id = model.config.label2id
        formatted_infer_l = label_fomatting(infer_label, label2id)
        infer_dataset = EMPDataset(infer_encodings, formatted_infer_l)

        evaluate(args, model, infer_dataset, mode, model_path_list[i].split('/')[-1][11:])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='모델 이름')
    parser.add_argument('--gpu_num', required=True, help='gpu 선택')
    parser.add_argument('--data_type', required=True, help='데이터셋 종류' )

    tmp_args = parser.parse_args()

    model_list = ['distilbert-base-uncased', "albert-xxlarge-v1", "albert-xxlarge-v2", 'roberta-large','bert-large-uncased', 'distilbert-base-uncased-distilled-squad']
    file_list = ['test.tsv']

    raw_test = read_data(file_list[0])
    test_t, test_l = read_data_split(raw_test)

    args = AttrDict()

    args.max_steps = -1
    args.gradient_accumulation_steps = 1
    args.num_train_epochs = 50
    args.learning_rate = 1e-5
    args.warmup_proportion = 0
    args.train_batch_size = 32
    args.eval_batch_size = 128
    args.logging_steps = 100
    args.save_steps = 100
    args.device = 'cuda:'+ tmp_args.gpu_num
    args.model_name = tmp_args.model#in model list
    args.output_dir = 'output/' + args.model_name
    args.max_grad_norm = 1.0
    args.evaluate_test_during_training = False
    args.save_optimizer = False

    result = inference(args, test_t, test_l, mode='test')
    print(result)
