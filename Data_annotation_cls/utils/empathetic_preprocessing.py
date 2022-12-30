import pandas as pd
import warnings
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거

file_list = ['../data/EMPD/valid.csv','../data/EMPD/test.csv','../data/EMPD/train.csv']

real_label = ['proud', 'caring', 'anxious', 'faithful', 'sentimental', 'surprised', 'sad', 'furious', 'trusting', 'annoyed', 'afraid', 'embarrassed', 'disgusted', 'angry', 'disappointed', 'lonely', 'anticipating', 'content', 'terrified', 'joyful', 'jealous', 'impressed', 'confident', 'excited', 'prepared', 'ashamed', 'guilty', 'apprehensive', 'devastated', 'grateful', 'hopeful', 'nostalgic']

label_dic = {}

for j in file_list:

    final = pd.DataFrame(columns=['sentence', 'label'])
    with open('{}'.format(j), 'r') as f:

        all_num = 0
        ck1 = 0
        prompt_set = set()
        memory = ""


        for i in tqdm(f.readlines()):
            if ck1 == 0:
                ck1 += 1
                continue


            all_num += 1
            tmp = i.split(',')[2:4]
            tmp[0], tmp[1] = tmp[1], tmp[0]

            if memory == "":
                memory = tmp[0]
            else:
                if tmp[0] == memory:
                    continue
                else:
                    memory = tmp[0]

            tmp[0] = tmp[0].replace('_comma_', ',')
            tmp[0] = tmp[0].replace('"', "")
            tmp[0] = tmp[0].replace('!', "")
            tmp[0] = tmp[0].replace('*', "")
            tmp[0] = tmp[0].replace('$', "")
            tmp[0] = tmp[0].replace('@', "")
            tmp[0] = tmp[0].replace('~', "")
            tmp[0] = tmp[0].replace('#', "")
            tmp[0] = tmp[0].replace('%', "")
            tmp[0] = tmp[0].replace('^', "")
            tmp[0] = tmp[0].replace('&', "")
            tmp[0] = tmp[0].replace('*', "")
            tmp[0] = tmp[0].replace('(', "")
            tmp[0] = tmp[0].replace(')', "")
            tmp[0] = tmp[0].replace('[', "")
            tmp[0] = tmp[0].replace(']', "")
            tmp[0] = tmp[0].replace('{', "")
            tmp[0] = tmp[0].replace('}', "")
            tmp[0] = tmp[0].replace('<', "")
            tmp[0] = tmp[0].replace('>', "")
            tmp[0] = tmp[0].replace('-', " ")
            tmp[0] = tmp[0].replace('/', " ")
            tmp[0] = tmp[0].lower()
            tmp[0] = tmp[0].strip() #, . 제외 특수문자 제거

            while tmp[0][-1] == " ":
                tmp[0] = tmp[0][:-1]

            if tmp[0][-1] != '.':
                tmp[0] = tmp[0] + '.'

            if tmp[1] not in label_dic.keys():
                label_dic[tmp[1]] = 1
            else:
                label_dic[tmp[1]] += 1

            tmp_dt = pd.DataFrame([tmp], columns=['sentence', 'label'])
            final = final.append(tmp_dt, ignore_index=True)
            prompt_set.add(tmp[0])

        final.reset_index(drop=True, inplace=True)
        final.to_csv('{}.tsv'.format(j[:-4]), index=False, sep='\t', encoding='utf-8')

        print('{} data prompt num : '.format(j), len(prompt_set))
        print('{} data num : '.format(j), all_num)

        if j == 'train.csv':
            for i in label_dic:
                print('{} : '.format(i), label_dic[i])


