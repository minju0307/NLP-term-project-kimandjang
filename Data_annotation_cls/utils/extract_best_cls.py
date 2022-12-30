import glob
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', required=True, help='모델 이름')
parser.add_argument('--data_type', required=True, help='데이터셋 종류')

args =parser.parse_args()

path = '../output/' + args.model_name
result_list = glob.glob(path+'/'+args.data_type +'/*.txt')
best_dic = { 'acc' : 0,
  'f1' : 0,
  'labels length' : 0,
  'precision' : 0,
  'recall' : 0,
  'weighted f1': 0,
}

best_step_dic ={}

for i in result_list:
    with open(i, 'r') as f:
        for j in f.readlines():
            j = j.strip()
            j = j.split('=')

            try:
                j[0] = j[0][:-1]
                j[1] = j[1][1:]

                if j[0] == 'acc':
                    if float(j[1]) > best_dic['acc'] :
                        best_dic['acc'] = float(j[1])
                        best_step_dic['acc'] = i.split('/')[-1]

                elif j[0] =='labels length':
                    if float(j[1]) > best_dic['labels length'] :
                        best_dic['labels length'] = float(j[1])
                        best_step_dic['label length'] = i.split('/')[-1]

                elif j[0] == 'precision' :
                    if float(j[1]) > best_dic['precision'] :
                        best_dic['precision'] = float(j[1])
                        best_step_dic['precision'] = i.split('/')[-1]

                elif j[0] == 'recall' :
                    if float(j[1]) > best_dic['recall'] :
                        best_dic['recall'] = float(j[1])
                        best_step_dic['recall'] = i.split('/')[-1]

                elif j[0] == 'f1' :
                    if float(j[1]) > best_dic['f1'] :
                        best_dic['f1'] = float(j[1])
                        best_step_dic['f1'] = i.split('/')[-1]

                else:
                    if float(j[1]) > best_dic['weighted f1'] :
                        best_dic['weighted f1'] = float(j[1])
                        best_step_dic['weigthed f1'] = i.split('/')[-1]
            except:
                continue
            finally:
                None
print()
print()
print()
print()

print(best_dic)
print(best_step_dic)

