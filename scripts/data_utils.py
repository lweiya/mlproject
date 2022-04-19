import configparser
import hashlib
import itertools
import json
import math
import os
import pickle
import random
import re
import time
import zipfile
import copy

import pandas as pd
import spacy
from doccano_api_client import DoccanoClient
from spacy.matcher import PhraseMatcher


def d_parse_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

configs = d_parse_config()



# 数据库导出字段 details
# doccano 导入字段 text label
# doccano 导出字段 data label
# label ： [[0,2,'O'],[1,3,'B-PER']]
# 分类标签 {"cats"：{"需要":1,"不需要":0}}

# 函数分为3个层次
# D Data manupulation 数据操作层
# P Preprocessing 数据预处理层
# B Building blocks 建构层

# 操作对象
# file - 直接的文件对象
# df - 数据表
# data - 列表

# ——————————————————————————————————————————————————
# 数据操作层
# ——————————————————————————————————————————————————

ROOT_PATH = '../'
ASSETS_PATH = ROOT_PATH + 'assets/'
DATA_PATH = ROOT_PATH + 'data/'
LOCK_FILE_PATH = ROOT_PATH + 'files_lock'
DATABASE_PATH = ROOT_PATH + 'database/'

# instantiate a client and log in to a Doccano instance
doccano_client = DoccanoClient(
    configs['doccano']['url'],
    configs['doccano']['user'],
    configs['doccano']['password']
)


# df保存jsonl文件
def d_save_df_datasets(df,path):
    with open(path,'w',encoding='utf-8') as f:
        for entry in df.to_dict('records'):
            json.dump(entry,f,ensure_ascii=False)
            f.write('\n')

# 数组保存jsonl文件
def d_save_list_datasets(data,path):
    with open(path,'w',encoding='utf-8') as f:
        for entry in data:
            json.dump(entry,f,ensure_ascii=False)
            f.write('\n')

# 读取数据集文件返回list
def d_read_json(path) -> list:
    data = []
    with open(path, 'r',encoding='utf8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 存储pkl
def d_save_pkl(value,path):
  with open(path, 'wb') as f:
    pickle.dump(value, f)

# 读取pkl
def d_read_pkl(path) -> object:
    with open(path, 'rb') as f:
        value = pickle.load(f)
    return value

# 写入平板文件，每行一个数据，以\n分隔
def d_save_file(files,path):
    with open(path, 'w') as f:
        for file in files:
            f.write(file + '\n')

# 读取平板文件，返回每行内容，去掉内容中的\n
def d_read_file(path) -> list:
    with open(path, 'r') as f:
        data = f.readlines()
    data = [i.replace('\n', '') for i in data]
    return data



# ——————————————————————————————————————————————————
# 数据处理层
# ——————————————————————————————————————————————————

# 生成md5
def p_generate_md5(text) -> str:
    m = hashlib.md5()
    m.update(text.encode('utf-8'))
    return m.hexdigest()

# 预处理html文本
def p_html_text(df,column):
    df[column] = df[column].apply(p_filter_tags)

# 返回新添加的文件
def p_get_new_files(lock_files, files) -> list:
    new_files = []
    for file in files:
        if file not in lock_files:
            new_files.append(file)
    return new_files

# 随机选择100个样本
def p_random_select(db,num=100) -> pd.DataFrame:
    db = db.sample(n=num)
    return db

# 获得文件和index
def p_get_data_index(db) -> pd.DataFrame:
    # 根据文件名获得ids，得到{"file_name":file_name,"id":[1,2,3,4,5]}
    db_selected = db.groupby('file_name').agg({'id':list})
    # 去掉index
    db_selected = db_selected.reset_index()
    return db_selected

# 抽取数据
def p_extract_data(db_selected) -> pd.DataFrame:
    # 循环db_selected，抽取数据
    dfs = []
    for idx,item in db_selected.iterrows():
        file_name = item['file_name']
        ids = item['id']
        # 抽取数据
        df = pd.read_csv('../assets/' + file_name)
        df = df.loc[ids]
        df['id'] = df.index
        df['file_name'] = file_name
        dfs.append(df)
    # 合并dfs
    df = pd.concat(dfs)
    return df

# 替换特殊字符
def p_replaceCharEntity(text) -> str:
    CHAR_ENTITIES = {'nbsp': ' ', '160': ' ',
                     'lt': '<', '60': '<',
                     'gt': '>', '62': '>',
                     'amp': '&', '38': '&',
                     'quot': '"', '34': '"', }

    re_charEntity = re.compile(r'&#?(?P<name>\w+);')
    sz = re_charEntity.search(text)
    while sz:
        entity = sz.group()  # entity全称，如&gt;
        key = sz.group('name')  # 去除&;后entity,如&gt;为gt
        try:
            text = re_charEntity.sub(CHAR_ENTITIES[key], text, 1)
            sz = re_charEntity.search(text)
        except KeyError:
            # 以空串代替
            text = re_charEntity.sub('', text, 1)
            sz = re_charEntity.search(text)
    return text

# 去掉网页标签
def p_filter_tags(text) -> str:
    # 先过滤CDATA
    re_cdata = re.compile('//<!\[CDATA\[[^>]*//\]\]>', re.I)  # 匹配CDATA
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)  # Script
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)  # style
    re_br = re.compile('<br\s*?/?>')  # 处理换行
    re_h = re.compile('</?\w+[^>]*>')  # HTML标签
    re_o = re.compile(r'<[^>]+>', re.S) # 其他
    re_comment = re.compile('<!--[^>]*-->')  # HTML注释
    s = re_cdata.sub('', text)  # 去掉CDATA
    s = re_script.sub('', s)  # 去掉SCRIPT
    s = re_style.sub('', s)  # 去掉style
    s = re_br.sub('\n', s)  # 将br转换为换行
    s = re_h.sub('', s)  # 去掉HTML 标签
    s = re_o.sub('',s)
    s = re_comment.sub('', s)  # 去掉HTML注释
    # 去掉多余的空行
    # blank_line = re.compile('\n+')
    # s = blank_line.sub('\n', s)
    s = p_replaceCharEntity(s)  # 替换实体
    return s

# 计算分割点
def p_cal_boder_end(block,length=500):
    borders_end = []
    # 计算出边界值
    for i in range(block):
        borders_end.append((i+1) * length)
    return borders_end

# 分割标签
def p_cal_border_and_label_belong(labels,borders_end):
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
def p_generate_new_datasets(new_data, text, labels, borders_end, borders_start, label_loc,id):
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

# ——————————————————————————————————————————————————
# 构建层
# ——————————————————————————————————————————————————


# 保存lockfile
def b_save_lock_file(file):
    d_save_file(file,path=LOCK_FILE_PATH)

# 读取lockfile
def b_read_lock_file() -> list:
    return d_read_file(path=LOCK_FILE_PATH)

# 拿到assets下面的文件列表
def b_read_assets() -> list:
    files = os.listdir(ASSETS_PATH)
    # 去掉.DS_Store
    files.remove('.DS_Store')
    return files

# 拿到data下面的文件列表
def b_read_data() -> list:
    files = os.listdir(DATA_PATH)
    # 去掉.DS_Store
    files.remove('.DS_Store')
    return files

# 找出新的文件
def b_get_new_files() -> list:
    lock_files = b_read_lock_file()
    files = b_read_data()
    new_files = p_get_new_files(lock_files, files)
    b_save_lock_file(files)
    return new_files

# 生成数据库入库文件，包括文件名，id,md5,清洁数据，并且根据md5去除重复的数据
def b_file_2_df(file_name,text_col='details') -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH + file_name)
    df['id'] = df.index
    df['file_name'] = file_name
    p_html_text(df,text_col)
    df['md5'] = df[text_col].apply(p_generate_md5)
    df['time'] = pd.to_datetime('now')
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S') + pd.Timedelta(hours=8)
    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df.rename(columns={text_col:'text'},inplace=True)
    df.drop_duplicates(subset=['md5'])
    return df

# 合并数据集
def b_combine_datasets(files:list) -> list:
    datas = []
    for file in files:
        data = d_read_json(ASSETS_PATH + file)
        datas.append(data)
    # 把列表的列表合并成一个列表
    return list(itertools.chain.from_iterable(datas))

# 统计label个数
def b_label_counts(data:list) -> dict:
    # 统计train当中label的个数
    label_count = {}
    for entry in data:
        label = entry['label']
        for item in label:
            label_ = item[2]
            if label_ in label_count:
                label_count[label_] += 1
            else:
                label_count[label_] = 1
    return label_count


# 保存所有数据
def b_save_db_all(df):
    d_save_pkl(df,DATABASE_PATH + 'all.pkl')

# 保存数据集分布
def b_save_db_datasets(df):
    d_save_pkl(df,DATABASE_PATH + 'datasets.pkl')

# 读取数据集分布
def b_read_db_datasets():
    return d_read_pkl(DATABASE_PATH + 'datasets.pkl')

# 保存清洗成后的数据
def b_save_db_basic(df):
    d_save_pkl(df,DATABASE_PATH + 'basic.pkl')

# 读取清洗好的数据
def b_read_db_basic():
    return d_read_pkl(DATABASE_PATH + 'basic.pkl')

# 将df保存为datasets
def b_save_df_datasets(df,file):
    d_save_df_datasets(df,ASSETS_PATH + file)

# 将data保存为datasets
def b_save_list_datasets(data,file):
    d_save_list_datasets(data,ASSETS_PATH + file)

# 读取最好ner模型
def b_load_best_model():
    return spacy.load("../training/model-best")

# 读取最好cats模型
def b_load_best_cats():
    return spacy.load("../training/cats/model-best")

# 读取最好test模型
def b_load_best_test():
    return spacy.load('../training/model-best-test')

# 读取文本文件到list当中
def b_read_text_file(file):
    data = d_read_file(ASSETS_PATH + file)
    return data

# 读取文本文件转换成为json到list当中
def b_read_dataset(file):
    data = d_read_json(ASSETS_PATH + file)
    return data

# 随机查看数据
def b_check_random(data,num):
    test = random.sample(data,num)
    for entry in test:
        labels = entry['label']
        text = entry['data']
        label = random.sample(labels,1)[0]
        print(text[label[0]:label[1]],label[2])



# 根据最好的模型、训练集，测试集生成cats模型
def b_generate_cats_datasets():
    train = b_read_dataset('train.json')
    dev = b_read_dataset('dev.json')

    # 合并训练集和测试集
    train_dev = train + dev

    # 提取data字段
    train_dev_data = [entry['data'] for entry in train_dev]

    nlp = b_load_best_model()

    docs = nlp.pipe(train_dev_data)

    predicts = []
    for doc in docs:
        predict = [[ent.text,ent.label_] for ent in doc.ents]
        predicts.append(predict)

    for sample,predict in zip(train_dev,predicts):
        text = sample['data']
        labels = sample['label']
        sample_label = [[text[label[0]:label[1]],label[2]] for label in labels]
        for entry in sample_label:
            if entry not in predict:
                sample['cats'] = {"需要":1,"不需要":0}
                break
            else:
                sample['cats'] = {"需要":0,"不需要":1} 
                break

    pos = []
    neg = []
    for sample in train_dev:
        if sample['cats']['需要'] == 1:
            pos.append(sample)
        else:
            neg.append(sample)

    # 随机排列pos，neg
    random.shuffle(pos)
    random.shuffle(neg)

    # train_cat,dev_cat
    train_cat = pos[:int(len(pos)*0.8)] + neg[:int(len(neg)*0.8)]
    dev_cat = pos[int(len(pos)*0.8):] + neg[int(len(neg)*0.8):]

    b_save_list_datasets(train_cat,'train_cats.json')
    b_save_list_datasets(dev_cat,'dev_cats.json')

# 读取训练集的标签情况
def b_read_train_label_counts():
    train = b_read_dataset('train.json')
    label_counts = b_label_counts(train)
    return label_counts

# 读取训练集的labels并且保存
def b_save_labes():
    label_counts = b_read_train_label_counts()
    l = list(label_counts.keys())
    d_save_file(l,"labels.txt")

# 分割数据集
def b_cut_datasets_size_pipe(file):
    data = b_read_dataset(file)

    new_data = []
    for entry in data:
        id = entry['id']
        text = entry['data']
        labels = entry['label']
        if len(text) < 500:
            new_data.append(entry)
        else:
            blcok = math.ceil(len(text) / 500)
            borders_end = p_cal_boder_end(blcok)
            borders_start = [0] + [i + 1 for i in borders_end[:-1]]
            label_loc = p_cal_border_and_label_belong(labels,borders_end)
            p_generate_new_datasets(new_data, text, labels, borders_end, borders_start, label_loc,id)
    return new_data

# 转换成百度excel格式，需要调整表头
def b_baidu_excel_format(file):
    new_data = b_cut_datasets_size_pipe(file)

    excel_data = []
    for entry in new_data:
        item = []
        item.append(entry['data'])
        labels = entry['label']
        for label in labels:
            loc = [label[0],label[1]]
            la = str(loc) +','+ label[2]
            item.append(la)
        excel_data.append(item)

    df = pd.DataFrame(excel_data)

    df.to_excel(file.split('.')[0] + '_new.xlsx',index=False)

# 读取新的文件进入到basic中
def b_add_new_file_to_db_basic(file):
    df = b_file_2_df(file)
    db = b_read_db_basic()
    df_db = pd.concat([db,df],axis=0)
    # 根据md5排重
    df_db = df_db.drop_duplicates(subset='md5',keep='first')
    b_save_db_basic(df_db)

# 从基础库中抽取未被业务未被抽样的数据
def b_extrct_data_from_db_basic(dataset_name) -> pd.DataFrame:
    db = b_read_db_basic()
    db_dataset = b_read_db_datasets()
    db_dataset = db_dataset[db_dataset['dataset'].str.contains(dataset_name)]
    return db[db['md5'].isin(db_dataset['md5']) == False]


# 根据cats模型选择数据
def b_select_data_by_model(dataset_name,num):
    db = b_extrct_data_from_db_basic('tender')
    nlp = b_load_best_cats()

    sample_data = []
    for index,row in db.iterrows():
        text = row['text']
        doc = nlp(text)
        if doc.cats['需要'] >= 0.5 and doc.cats['需要'] <= 0.6:
            sample_data.append(row)
        if len(sample_data) == 100:
            break

    return pd.DataFrame(sample_data)

# 上传到doccano测试项目
def b_doccano_upload(file,project_id):
    doccano_client.post_doc_upload(project_id,file,ASSETS_PATH)

    # 从doccano获取数据
def b_doccano_export_project(project_id,path):
    url = configs['doccano']['url']
    result = doccano_client.post(f'{url}/v1/projects/{project_id}/download', json={'exportApproved': False, 'format': 'JSONL'}) 
    task_id = result['task_id']
    while True:
        result = doccano_client.get(f'{url}/v1/tasks/status/{task_id}')
        if result['ready']:
            break
        time.sleep(1)
    result = doccano_client.get_file(f'{url}/v1/projects/{project_id}/download?taskId={task_id}')
    tmp_zip_path = ASSETS_PATH + '1.zip'
    with open(tmp_zip_path, 'wb') as f:
        for chunk in result.iter_content(chunk_size=8192): 
            f.write(chunk)
    zipfile.ZipFile(tmp_zip_path).extractall(path=ASSETS_PATH)
    os.rename(ASSETS_PATH + 'all.jsonl', ASSETS_PATH + path)
    os.remove(tmp_zip_path)


# 删除项目中的数据
def b_doccano_delete_project(project_id):
    r = doccano_client.get_document_list(project_id)
    length = r['count']
    r = doccano_client.get_document_list(project_id,{'limit':[length],'offset':[0]})
    for entry in r['results']:
        doccano_client.delete_document(project_id,entry['id'])

# 在doccnano中查看某个标签的情况
# b_doccano_dataset_label_view('train.json',['招标项目编号'],1)
def b_doccano_dataset_label_view(file,labels,project_id):
    b_doccano_delete_project(project_id)
    train = b_read_dataset(file)
    new_train = []
    for entry in train:
        text = entry['data']
        for label in entry['label']:
            new_entry = copy.deepcopy(entry)
            new_entry.remove('data')
            new_entry['id'] = entry['id']
            start = label[0]
            end = label[1]
            s_start = start - 200 if start - 200 > 0 else 0
            s_end = end + 200 if end + 200 < len(text) else len(text)
            new_entry['text'] = text[s_start:s_end]
            label_ = [[start - s_start,end - s_start,label[2]]]
            new_entry['label'] = label_
            new_entry['s_start'] = s_start
            new_entry['s_end'] = s_end
            if label[2] in labels:
                new_train.append(new_entry)
    b_save_list_datasets(new_train,'train_new.json')
    b_doccano_upload('train_new.json',project_id)   

# b_cat_data_to_doccano(db,100,['招标编号','招标项目编号'])
# 找出一些关键词的数据
def b_doccano_cat_data(df,number,terms):

    number = number

    new_trian = []

    nlp = spacy.load('zh_core_web_lg')
    matcher = PhraseMatcher(nlp.vocab)
    terms = terms
    patterns = [nlp.make_doc(text) for text in terms]
    matcher.add("匹配条件", patterns)

    for index,row in df.iterrows():
        new_entry = row
        text = row['text']
        doc = nlp(text)
        matches = matcher(doc)
        labels = []
        for match_id, start, end in matches:
            span = doc[start:end]
            start = span.start_char
            end = span.end_char
            label =[start,end,'其他']
            labels.append(label)
        new_entry['label'] = labels
        new_trian.append(new_entry)
        if len(new_trian) == number:
            break
    df = pd.DataFrame(new_trian)
    df.rename(columns={'text':'data'},inplace=True)
    b_save_df_datasets(df,'train_cat.json')
    b_doccano_dataset_label_view('train_cat.json',['其他'],1)

# 随机根据业务初始化数据集，并且上传到doccano
def b_doccano_init_dataseet(name,num,ratio,train_id,test_id):
    db = b_read_db_basic()
    # 随机抽取1000条数据
    df_db = pd.DataFrame(db)
    df_db = df_db.sample(num)

    # 按照这2：8的比例切分训练和测试
    df_train,df_test = b_split_train_test(df_db,ratio)

    # 分别保存到json文件中
    b_save_df_datasets(df_train,'train.json')
    b_save_df_datasets(df_test,'test.json')

    # 分别上传到doccano
    b_doccano_upload('train.json',train_id)
    b_doccano_upload('test.json',test_id)

    df_train['dataset'] = name + '_train'
    df_test['dataset'] = name + '_test'

    # 合并两个数据集
    df_train_test = pd.concat([df_train,df_test])

    b_save_db_datasets(df_train_test)

# 标注数据集
# b_label_dataset
def b_label_dataset(file):
    file_name = file.split('.')[0]
    data = b_read_dataset(file)
    nlp = b_load_best_model()
    data_text = [ entry['text'] for entry in data ]
    docs = nlp.pipe(data_text)
    for doc,sample in zip(docs,data):
        labels = []
        for ent in doc.ents:
            labels.append([ent.start_char,ent.end_char,ent.label_])
        sample['label'] = labels
    b_save_list_datasets(data,file_name + '_label.json')


# 把demo的label更新到原来的file中
# b_change_label('train_dev.json','train_dev_label.json',['招标项目编号'])
def b_change_label(file,label_file,label_names):
    b_doccano_export_project(1,label_file)

    data = b_read_dataset(file)
    label_data = b_read_dataset(label_file)

    for label_sample in label_data:
        id = label_sample['id']
        s_start = label_sample['start']
        for sample in data:
            if sample['id'] == id:
                break
        for idx,label in enumerate(sample['labels']):
            if label[2] in label_names:
                sample['labels'].remove(label)
        for label in label_sample['label']:
            if label[2] in label_names:
                start = label[0] + s_start
                end = label[1] + s_start
                sample['labels'].append([start,end,label[2]])
    
    b_save_list_datasets(data,file)
    
# 转换json变成bio
def b_json2bio(file):
    '''
    将json文件中的数据转录为BIO形式，保存规则可以在43行修改
    '''
    file_name = file.split('.')[0]
    f_write = open(ASSETS_PATH + file_name + '_bio.txt', 'w', encoding='utf-8')
    load = b_read_dataset(file)
    for i in range(len(load)):
        labels = load[i]['label']
        text = load[i]['text']
        tags = ['O'] * len(text)
        for j in range(len(labels)):
            label = labels[j]
            tags[label[0]] = 'B-' + str(label[2])
            k = label[0]+1
            while k < label[1]:
                tags[k] = 'I-' + str(label[2])
                k += 1
        print(tags)
        for word, tag in zip(text, tags):
            f_write.write(word + '\t' + tag + '\n')
        f_write.write("\n")

# 将预测转换成json
def b_convert_bio_json(text,predict):
    string="我是李明，我爱中国，我来自呼和浩特"
    predict=["o","o","i-per","i-per","o","o","o","b-loc","i-loc","o","o","o","o","b-per","i-loc","i-loc","i-loc"]
    item = {"string": string, "entities": []}
    entity_name = ""
    flag=[]
    visit=False
    for char, tag in zip(string, predict):
        if tag[0] == "b":
            if entity_name!="":
                x=dict((a,flag.count(a)) for a in flag)
                y=[k for k,v in x.items() if max(x.values())==v]
                item["entities"].append({"word": entity_name,"type": y[0]})
                flag.clear()
                entity_name=""
            entity_name += char
            flag.append(tag[2:])
        elif tag[0]=="i":
            entity_name += char
            flag.append(tag[2:])
        else:
            if entity_name!="":
                x=dict((a,flag.count(a)) for a in flag)
                y=[k for k,v in x.items() if max(x.values())==v]
                item["entities"].append({"word": entity_name,"type": y[0]})
                flag.clear()
            flag.clear()
            entity_name=""
    
    if entity_name!="":
        x=dict((a,flag.count(a)) for a in flag)
        y=[k for k,v in x.items() if max(x.values())==v]
        item["entities"].append({"word": entity_name,"type": y[0]})
    return item 

# 传入df，划分数据集
def b_split_train_test(df_db,ratio):
    df_train = df_db.sample(frac=ratio)
    df_test = df_db.drop(df_train.index)
    return df_train,df_test



# 从labels.txt中生成biolabels
# b_generate_biolabels_from('labels.txt')
def b_generate_biolabels_from(file):
    labels = b_read_text_file(file)

    bio_labels = []
    for label in labels:
        bio_labels.append('B-'+label)
        bio_labels.append('I-'+label)
    bio_labels = ['O'] + bio_labels
    return bio_labels


# 将json标注改成数组对应标注法
# b_trans_dataset_bio(bio_labels,'train.json')
def b_trans_dataset_bio(bio_labels,file):
    file_name = file.split('.')[0]
    data  = b_read_dataset(file)
    
    new_data = []
    for sample in data:
        new_sample = {}
        text = sample['data']
        l_text = list(text)
        new_sample['data'] = l_text
        new_sample_label = [0] * len(l_text)
        for label in sample['label']:
            start = label[0]
            end = label[1]
            label_ = label[2]
            new_sample_label[start] = bio_labels.index('B-' + label_) 
            for i in range(start+1,end):
                new_sample_label[i] = bio_labels.index('I-' + label_)
        new_sample['label'] = new_sample_label
        new_data.append(new_sample)

    b_save_list_datasets(new_data,file_name + '_trf.json')

# 根据datasource查找原始数据
def b_find_orig_by_data_source(file,data_source):
    df = pd.read_csv(DATA_PATH + file)
    return df[df['data_source']==data_source].values[0]['details']

# doccano导出数据中不为空的数据整理出来
# b_get_all_label_data('train.jsonl')
def b_get_all_label_data(file):
    file_name = file.split('.')[0]

    train = b_read_dataset(file)

    new_train = []

    for sample in train:
    # 把label长度不为空的数据提取出来
        if len(sample['label']) != 0:
            new_train.append(sample)

    b_save_list_datasets(new_train,file_name + '.json')

# 去除标签中的空格字符
# b_remove_invalid_label('train.json')
def b_remove_invalid_label(file):
    invalid_span_tokens = re.compile(r'\s')

    data = b_read_dataset(file)

    cleaned_datas = []
    for sample in data:
        cleaned_data = {}
        text = sample['data']
        cleaned_data['data'] = text
        labels = sample['label']
        clean_labels = []
        for start,end,label in labels:
            valid_start = start
            valid_end = end
        # if there's preceding spaces, move the start position to nearest character
            while valid_start < len(text) and invalid_span_tokens.match(
                text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                text[valid_end - 1]):
                valid_end -= 1
            clean_labels.append([valid_start, valid_end, label])
        cleaned_data['label'] = clean_labels
        cleaned_datas.append(cleaned_data)  

    b_save_list_datasets(cleaned_datas,'file')

# 把bio数据集划分成最长的数据集,并且保存为train_trf_max.json
#split_dataset_by_max('train_trf.json',510) 
def b_bio_split_dataset_by_max(file,max_len):
    file_name = file.split('.')[0]
    max_length = max_len

    data = b_read_dataset('dev_trf.json')

    new_data = []
    for sample in data:
        data_text = sample["data"]
        data_label = sample["label"]

        divs = len(data_text)/max_length + 1

        every = len(data_text)// divs

        befor_after = (max_length - every ) // 2


        for i in range (0,int(divs)):
            new_sample = {}
            start  = i * every
            end = (i+1) * every
            if i == 0:
                end = end + befor_after * 2
            elif i == int(divs) - 1:
                start = start - befor_after * 2
            else:
                start = start - befor_after
                end = end + befor_after
            start = start if start >= 0 else 0
            end = end if end <= len(data_text) else len(data_text)
            start = int(start)
            end = int(end)
            new_text_data = data_text[start:end]
            new_label_data = data_label[start:end]
            new_sample["data"] = new_text_data
            new_sample["label"] = new_label_data
            new_data.append(new_sample)

    b_save_list_datasets(new_data,file_name  + '_maxlen.json')


# 同步train.json和dev.json的数据到doccano中
def b_doccano_train_dev():
    train = b_read_dataset('train.json')
    dev = b_read_dataset('dev.json')

    train_dev = train + dev

    df = pd.DataFrame(train_dev)

    df['md5'] = df['data'].apply(p_generate_md5)

    db = b_read_db_datasets()

    db_new = pd.merge(db,df,left_on='md5',right_on='md5',how='left')

    db_new = db_new.dropna()

    db_new = db_new.drop(['data'],axis=1)

    db_new_train = db_new[db_new['dataset']=='tender_train']
    db_new_dev = db_new[db_new['dataset']=='tender_dev']

    b_save_df_datasets(db_new_train,'train_imp.json')
    b_save_df_datasets(db_new_dev,'dev_imp.json')

    b_doccano_upload('train_imp.json',2)
    b_doccano_upload('dev_imp.json',3)

    # 去掉db_new的label列
    db_new = db_new.drop(['label'],axis=1)

    b_save_db_datasets(db_new)

# 合并train_dev数据，并且附加上meta，保存到train_dev.json
def b_combine_train_dev_meta():
    train = b_read_dataset('train.json')
    dev = b_read_dataset('dev.json')

    train_dev = train + dev

    df = pd.DataFrame(train_dev)

    db = b_read_db_datasets()

    df['md5'] = df['data'].apply(p_generate_md5)

    db_new = pd.merge(db,df,left_on='md5',right_on='md5',how='left')

    db_new = db_new.dropna()

    db_new.rename(columns={'data':'text'},inplace=True)

    b_save_df_datasets(db_new,'train_dev.json')



# ——————————————————————————————————————————————————
# 调用
# ——————————————————————————————————————————————————

if __name__ == '__main__':
    pass



