import torch
import glob
from transformers import DistilBertTokenizerFast, AlbertTokenizerFast, BertTokenizerFast, RobertaTokenizerFast
from transformers import DistilBertForSequenceClassification, AlbertForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification
from fastprogress.fastprogress import master_bar, progress_bar
import logging
logger = logging.getLogger(__name__)
from attrdict import AttrDict
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
import os
from sklearn import metrics as sklearn_metrics
import numpy as np

class EMPDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def read_data(raw_data):
    whole = []
    with open("data/"+raw_data, 'r') as f:
        for i in f.readlines():
            tmp = i.split('\t')
            tmp[1] = tmp[1].strip()
            whole.append(tmp)
        whole = whole[1:]
        return whole


def read_data_split(data):
    texts = []
    label = []
    for i in data:
        texts.append(i[0])
        label.append(i[1])
    return texts, label


def create_label_dic(t_label):
    label_dic = {}
    for i in list(set(t_label)):
        label_dic[i] = list(set(t_label)).index(i)

    return label_dic


def label_fomatting(label_data, label_dic):
    tmp_data =[]
    for i in range(len(label_data)):
        tmp_data.append(label_dic[label_data[i]])

    return tmp_data


def simple_accuracy(labels, preds):
    return (labels == preds).mean()


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)

    return {
        "acc": simple_accuracy(labels, preds),
        "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
        "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
        "weighted f1": sklearn_metrics.f1_score(labels, preds, average="weighted"),
        "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        'labels length': len(labels),
    }

def train(args,
          model,
          train_dataset,
          dev_dataset=None,
          test_dataset=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion),
                                                num_training_steps=t_total)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()
    mb = master_bar(range(int(args.num_train_epochs)))

    for epoch in mb:

        epoch_iterator = progress_bar(train_dataloader, parent=mb)

        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(batch[t].to(args.device) for t in batch)

            if args.model_name in ['roberta-large', 'distilbert-base-uncased','distilbert-base-uncased-distilled-squad']:
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[2]
                }
            else:
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[2],
                    'labels': batch[3]
                }

            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_test_during_training:
                        evaluate(args, model, test_dataset, "test", global_step)
                    else:
                        evaluate(args, model, dev_dataset, "dev", global_step)

                if args.save_steps > 0 and global_step % args.save_steps == 0:

                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to {}".format(output_dir))

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        mb.write("Epoch {} done".format(epoch + 1))

        if args.max_steps > 0 and global_step > args.max_steps:
            break
    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset, mode, global_step=None):

    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if global_step != None:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(batch[t].to(args.device) for t in batch)

        with torch.no_grad():
            if args.model_name in  ['roberta-large', 'distilbert-base-uncased','distilbert-base-uncased-distilled-squad']:
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[2]
                }
            else:
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[2],
                    'labels': batch[3]
                }


            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()

        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    result = compute_metrics(out_label_ids, preds)
    results.update(result)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir,
                                    "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))

    print('eval done .')

    return results


if __name__ == '__main__':

    model_list = ['distilbert-base-uncased', "albert-xxlarge-v1", "albert-xxlarge-v2", 'roberta-large', "bert-large-uncased",'distilbert-base-uncased-distilled-squad', 'distilgpt2']

    args = AttrDict()

    args.max_steps = -1
    args.gradient_accumulation_steps = 1
    args.num_train_epochs = 20
    args.learning_rate = 1e-5
    args.warmup_proportion = 0
    args.train_batch_size = 32
    args.eval_batch_size = 128
    args.logging_steps = 100
    args.save_steps = 100
    args.device = 'cuda:2'
    args.model_name = 'distilbert-base-uncased-distilled-squad'  # in model_list
    args.output_dir = 'output/' + args.model_name
    args.max_grad_norm = 1.0
    args.evaluate_test_during_training = False
    args.save_optimizer = False

    file_list = ['train.tsv','valid.tsv', 'test.tsv']
    #---------
    raw_train = read_data(file_list[0])
    raw_valid = read_data(file_list[1])
    raw_test = read_data(file_list[2])

    train_t, train_l = read_data_split(raw_train)
    valid_t, valid_l = read_data_split(raw_valid)
    test_t, test_l = read_data_split(raw_test)

    #-----------
    tokenizer =DistilBertTokenizerFast.from_pretrained(args.model_name)

    train_encodings = tokenizer(train_t, truncation=True, padding=True)
    val_encodings = tokenizer(valid_t, truncation=True, padding=True)
    test_encodings = tokenizer(test_t, truncation=True, padding=True)
    #------------

    label2id = create_label_dic(train_l)

    formatted_train_l = label_fomatting(train_l, label2id)
    formatted_valid_l = label_fomatting(valid_l, label2id)
    formatted_test_l = label_fomatting(test_l, label2id)
    #------------- finally, using encoded sentences data and formatted label data (_encodings, formatted_l)

    train_dataset =  EMPDataset(train_encodings, formatted_train_l)
    val_dataset = EMPDataset(val_encodings, formatted_valid_l)
    test_dataset = EMPDataset(test_encodings, formatted_test_l)

    model = DistilBertForSequenceClassification.from_pretrained(args.model_name,
                                                                id2label = {str(i) : label for label, i in label2id.items() },
                                                                label2id = label2id,
                                                                num_labels=32)

    if torch.cuda.is_available():
        model.to(args.device)

    global_step, tr_loss = train(args, model, train_dataset, val_dataset, test_dataset)

