# 计算分割点
def cal_boder_end(block,length=4800):
    borders_end = []
    # 计算出边界值
    for i in range(block):
        borders_end.append((i+1) * length)
    return borders_end

# 分割标签
def cal_border_and_label_belong(labels,borders_end):
    label_loc = []
    for label in labels:
        start = label[0]
        end = label[1]
        for idx,border in enumerate(borders_end):
            if start < border and end < border:
                label_loc.append(idx)
                break
            if start < border and end > border:
                pad = end - border + 20
                label_loc.append(idx)
                for idxc,border in enumerate(borders_end):
                    if idxc >= idx:
                        borders_end[idxc] += pad
                break
    return label_loc

# 拆分数据集
def generate_new_datasets(new_data, text, labels, borders_end, borders_start, label_loc,id):
    idx = 0
    for b_start,b_end in zip(borders_start,borders_end):
        entry = {}
        entry['data'] = text[b_start:b_end]
        new_labels = []
        for idxl,loc in enumerate(label_loc):
            if loc == idx:
                label = labels[idxl].copy()
                label[0] -= b_start
                label[1] -= b_start
                new_labels.append(label)
        entry['label'] = new_labels
        entry['id'] = id
        idx += 1
        if len(new_labels) != 0:
            new_data.append(entry)

# 分割数据集
def cut_datasets_size_pipe():
    data = read_datasets('train.json')

    new_data = []
    for entry in data:
        id = entry['id']
        text = entry['data']
        labels = entry['label']
        if len(text) < 4800:
            new_data.append(entry)
        else:
            blcok = math.ceil(len(text) / 4800)
            borders_end = cal_boder_end(blcok)
            borders_start = [0] + [i + 1 for i in borders_end[:-1]]
            label_loc = cal_border_and_label_belong(labels,borders_end)
            generate_new_datasets(new_data, text, labels, borders_end, borders_start, label_loc,id)

    df = pd.DataFrame(new_data)
    save_datasets(df,file_name='new_train.json',is_label=True)

    # 合并训练集和测试集
def combin_datasets(train_file="train.json",dev_file="dev.json"):
    # 读取train.json
    train = read_datasets(train_file)
    # 读取dev.json
    dev = read_datasets(dev_file)
    # 合并
    train.extend(dev)
    return train


def train_data_evaluate():
    train = combin_datasets(train_file="train.json",dev_file="dev.json")

    nlp = spacy.load('../training/model-best')

    text_data = [entry['data'] for entry in train]

    docs = nlp.pipe(text_data)

    for idx,doc in enumerate(docs):
        entry = train[idx]
        label_p = [[ent.text,ent.start,ent.end,ent.label_] for ent in doc.ents]
        label = entry['label']
        new_labels = []
        for item in label:
            new_label = []
            start = item[0]
            end = item[1]
            label_ = item[2]
            text = entry['data'][start:end]
            new_label.append(text)
            new_label.append(start)
            new_label.append(end)
            new_label.append(label_)
            new_labels.append(new_label)
        entry['label_p'] = label_p
        entry['new_label'] = new_labels
    return train


def label_right_counts(train):
    # 统计train当中label_p标注正确的个数
    label_p_count = {}
    for entry in train:
        label_p = entry['label_p']
        new_label = entry['new_label']
        for item in label_p:
            label_ = item[3]
            text = item[0]
            for new_item in new_label:
                if new_item[0] == text and new_item[3] == label_:
                    if label_ in label_p_count:
                        label_p_count[label_] += 1
                    else:
                        label_p_count[label_] = 1
    return label_p_count

def label_acc(label_count, label_p_count):
    # 通过label_count 和 label_p_count 计算准确率
    label_accuracy = {}
    for label_,count in label_count.items():
        if label_ in label_p_count:
            label_accuracy[label_] = label_p_count[label_] / count
        else:
            label_accuracy[label_] = 0
    return label_accuracy

df = b_file_2_df('Untitled.csv',text_col="details")

    # 保存数据库
    b_save_database_all(df)

    # 选择所有的数据
    df = p_get_data_index(df)

    # 提取数据
    df = p_extract_data(df)

    # 预处理
    p_html_text(df,'details')

    # 提取details中包含 “招标编号”的数据
    df_t = df[df['details'].str.contains('招标编号')]

    # 随机调整顺序
    df_t = df_t.sample(frac=1)

    # 抽取500个数据
    df_t = df_t.head(500)

    # 新增列dataset，前面50个为 tender_dev , 后面 450个为 tender_train
    df_t['dataset'] = 'tender_train'
    df_t.iloc[0:50,df_t.columns.get_loc('dataset')] = 'tender_dev'

    # 统计 dataset 个数
    df_t['dataset'].value_counts()

    # 将details改名为text
    df_t = df_t.rename(columns={'details':'text'})

    # 保存数据集
    tender_train  = df_t[df_t['dataset'] == 'tender_train']
    tender_dev = df_t[df_t['dataset'] == 'tender_dev']

    
    # 保存数据集
    with open('../assets/tender_train.json','w',encoding='utf-8') as f:
        for entry in tender_train.to_dict('records'):
            json.dump(entry,f,ensure_ascii=False)
            f.write('\n')

    # 保存测试集
    with open('../assets/tender_dev.json','w',encoding='utf-8') as f:
        for entry in tender_dev.to_dict('records'):
            json.dump(entry,f,ensure_ascii=False)
            f.write('\n')

    # 去掉text
    df_t = df_t.drop(columns=['text'])

    # 保存数据库
    b_save_database_datasets(df_t)


    def cut_sent(txt):
    txt = re.sub('([。])',r"\1\n",txt) # 单字符断句符，加入中英文分号
    txt = txt.rstrip()       # 段尾如果有多余的\n就去掉它
    nlist = txt.splitlines()
    nlist = [x for x in nlist if x.strip()!='']  # 过滤掉空行
    return nlist

# 读取训练文件
data =b_read_dataset('train.json')


ret = []
for entry in data:
    txt = entry['data']
    # 分句
    sents = cut_sent(txt)
    total = len(sents)
    # 计算每句的长度
    sents_segment = list(map(len, sents))
    # 累加得到每句的起止位置
    pos = np.cumsum(sents_segment)
    sents_pos = list( zip( [0] + pos.tolist(), pos))


    # 读取标签的位置
    labels = entry['label']
    labels_pos = [x[:2] for x in labels]
    # 按位置排序一下
    labels_pos = sorted(labels_pos, key=lambda x:x[0])


    labels_count = len(labels_pos)


    # 遍历两个位置，去掉没有重叠的部分
    i,j=0,0
    spos = sents_pos[i]
    lpos = labels_pos[j]
    result = []
    retdat = {}
    while 1:
        # 判断spos 和 lpos 的交叉情况
        if lpos[1]<=spos[0]: # 标签在句子左，继续移动标签
            j+=1
        if spos[1]<=lpos[0]: # 句子在标签左，继续移动句子
            i+=1

        if spos[0]<=lpos[1]<=spos[1] :
            # 加字典的方式返回
            if i in retdat.keys():
                retdat[i].append((lpos,j))
            else:
                retdat[i] = [(lpos,j)]
            j+=1
            
        if lpos[0]<=spos[1]<=lpos[1] :
            #result.append([spos,lpos,i,j])
            if i in retdat.keys():
                retdat[i].append((lpos,j))
            else:
                retdat[i] = [(lpos,j)]
            i+=1
            #j+=1
        if i>=total: break
        if j>=labels_count: break
        spos = sents_pos[i]
        lpos = labels_pos[j]
    

    for k,v in retdat.items():
        sb,se = sents_pos[k] # 句子位置
        sent_txt = txt[sb:se]
        sub_labels = []
        for (lb,le), j in v:
            lbltxt = txt[lb:le]
            lbl_label = labels[j][2]
            sub_labels.append([lb-sb,le-sb,lbl_label])

        ret.append({'data': sent_txt, 'label':sub_labels})

b_save_list_datasets(ret,'train.json')



