from src.util import processOutput, tensor_to_str, load_batch_iterator_with_eos, get_classifier
from torchtext.legacy.data import TabularDataset
from src.model import *
import json, logging, argparse

logger = logging.getLogger("train_tsf_model.py")

parser = argparse.ArgumentParser(description='Argparse for training transfer model')
parser.add_argument('--embedding-size', type=int, default=128,
                    help='yelp set to 128')
parser.add_argument('--hidden-size', type=int, default=500,
                    help='hidden size set to 500')
parser.add_argument('--batch-size', type=int, default=512,
                    help='batch size set to 512 for yelp')
parser.add_argument('--attn-size', type=int, default=100,
                    help='attn size set to 512 for yelp')
parser.add_argument('--data', type=str, default="Conv",
                    help='data')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--seed', type=int, default=0000,
                    help='random seed')
parser.add_argument('--max-epoch', type=int, default=40,
                    help='max epoch')
parser.add_argument('--log_iter', type=int, default=100,
                    help='log iteration')
parser.add_argument('--eval_iter', type=int, default=100,
                    help='log iteration')
parser.add_argument('--styleLossCoef', type=float, default=0.15,
                    help='styleLossCoefficient')
parser.add_argument('--style_size', type=int, default=200,
                    help='style vector size')
config = parser.parse_args()

config.data = 'Convai2'
config.name =  "{}-{}".format(config.data, config.styleLossCoef)
config.folder_name = config.name + 'real_v3'
config.cp = 47000

if torch.cuda.is_available():
    config.device = "cuda"
else:
    config.device = "cpu"

config.num_class = 32
config.data_path = f"data/{config.data}"

bleu_weight = (0.25, 0.25, 0.25, 0.25)

# Load Dataset
train, dev, test, train_iter, dev_iter, test_iter, X_VOCAB, C_LABEL = load_batch_iterator_with_eos(config.data_path, train="train.jsonl", val="dev.jsonl", test = "infer.jsonl", batch_size=config.batch_size,device=config.device)

# Model Setting
input_size = len(X_VOCAB.vocab)
pad_idx = X_VOCAB.vocab.stoi["<pad>"]

# Load Models
enc_cls, attn_cls, senti_cls = get_classifier(input_size, config.embedding_size, config.attn_size , config.hidden_size , pad_idx, config.device, config.data, "cls")
enc_r, attn_r, senti_r= get_classifier(input_size, config.embedding_size, config.attn_size , config.hidden_size , pad_idx, config.device, config.data, "r")

enc = EncoderRNN(input_size, config.embedding_size, config.hidden_size, pad_idx).to(config.device)
dec = DecoderRNN_LN(vocab_size = len(X_VOCAB.vocab), max_len=None, hidden_size = config.hidden_size+config.style_size, sos_id=X_VOCAB.vocab.stoi["<bos>"], eos_id=X_VOCAB.vocab.stoi["<eos>"], pad_id=X_VOCAB.vocab.stoi["<pad>"], style_size=config.style_size, num_class=config.num_class, use_attention=False).to(config.device)

#load parameters model parameters

dec.load_state_dict(torch.load("./model/{}/{}_dec.pth".format(config.folder_name, config.cp)))
enc.load_state_dict(torch.load("./model/{}_enc_r.pth".format(config.data)))

#load parameters Generator

G = Generator(enc_cls, attn_cls, enc).to(config.device)
G.load_state_dict(torch.load('./model/{}/{}_G.pth'.format(config.folder_name, config.cp)))

test_ = TabularDataset(path="./data/Conv/infer.jsonl",
                  format="json",
                  fields={"X":("X", X_VOCAB),
                          "C": ('C', C_LABEL)})


def getLossAndMetrics(iter,G, dec, config, pad_idx, enc_cls, attn_cls, senti_cls, X_VOCAB, C_LABEL, save=True):
    acc, total, bleu_scores = 0, 0, 0
    sentences=[]
    labels = []
    real_sentences= []
    real_labels = []
    self_loss_list, style_loss_list, mse_loss_list, cycle_loss_list = 0,0,0,0

    for i, batch in enumerate(iter):

        max_len = batch.X[1].max().cpu()
        hidden, output, attn_hidden, src_mask, _ = G(batch.X)
        _, _, output_dictionary = dec(None, hidden.unsqueeze(0), output, mask = src_mask, attn_hidden = attn_hidden, label=batch.C, gumbel=False, max_length =max_len)
        length, soft_text, text, srcmask = processOutput(output_dictionary, pad_idx, config.device)

        output, hidden = enc_cls(soft_text)
        scores, attn_hidden, reverse_scores, src_mask = attn_cls(output, soft_text[1])
        logits = senti_cls(attn_hidden)
        _, preds = logits.max(dim=-1)
        correct = (preds==batch.C).sum().item()
        total += logits.size(0)
        acc += correct
        labels.extend((batch.C).tolist())
        real_labels.extend((batch.C).tolist())

        for t, leng, real, real_len in zip(text.T, length, batch.X[0].T, batch.X[1]):
            pred = tensor_to_str(t[:int(leng.item()-1)],X_VOCAB).lower()
            candi = tensor_to_str(real[:int(real_len.item() - 1)], X_VOCAB).lower()

            real_sentences.append(candi)
            sentences.append(pred)

    # for i, line in enumerate(zip(sentences, labels)):
    #     print('index :', i, 'sentences : ', line)


    if save:
        with open('output/{}/infer.jsonl'.format(config.folder_name), 'w') as fp: #output 파일 이름
            for i, line in enumerate(zip(sentences, labels)):
                #print('index :', i, 'sentences : ', line)
                data_json={}
                data_json["index"] = i
                data_json["X"] = line[0].replace("\n","")
                data_json["C"] = C_LABEL.vocab.itos[line[1]]
                json.dump(data_json, fp)
                fp.write("\n")

        with open('output/{}/True_infer.jsonl'.format(config.folder_name), 'w') as fp: #output 파일 이름
            for i, line in enumerate(zip(real_sentences, real_labels)):
                print('index :', i, 'sentences : ', line)
                data_json={}
                data_json["index"] = i
                data_json["X"] = line[0].replace("\n","")
                data_json["C"] = C_LABEL.vocab.itos[line[1]]
                json.dump(data_json, fp)
                fp.write("\n")

    pass

# with open('output/{}_true_dev.jsonl'.format(config.name), 'w') as fp:  # dev파일 저장
#     for i, line in enumerate(zip(sentences, labels)):
#         data_json = {}
#         data_json["index"] = i
#         data_json["X"] = line[0].replace("\n", "")
#         data_json["C"] = C_LABEL.vocab.itos[line[1]]
#         json.dump(data_json, fp)
#         fp.write("\n")


getLossAndMetrics(test_iter,G, dec, config, pad_idx, enc_cls, attn_cls, senti_cls, X_VOCAB, C_LABEL, save=True)

