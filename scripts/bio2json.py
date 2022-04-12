import json


def list_of_groups(init_list, children_list_len):
    """把一个列表按指定数目分成多个列表,children_list_len是你指定的子列表的长度"""
    list_of_groups = zip(*(iter(init_list),) * children_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % children_list_len
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list

def transform_txt_json(txtdir,jsondir):
    """
    :param txtdir: 模型预测输出bio文件路径
    :param jsondir: 转换后json文件存放路径
    """
    f = open(txtdir,'r',encoding='utf-8')        # 模型预测输出bio文件路径
    output = open(jsondir, 'a', encoding='utf-8')     # 转换后json文件存放路径
    text = []
    labels = []
    labelsx = []
    lab = labn = []
    i = n = ln = 0
    start = end = 0     # 每个标签对应的起始位置
    for line in f.readlines():
        if line != "\n":
            str = list(line)     # 将每一行的字符串转换成以单字符为单位的列表
            text.append(str[0])  # 将第一个字符输入到text
            if str[2] == "B":
                lab = []
                start = i
                if n < start:
                    labelsx.append(n)
                    labelsx.append(end)
                    labelsx.append(labn)        # 将标签的起始-终止位置以及对应标签传入labelsx
                n = start
                end = i+1
                for k in range(4,len(str)-1):     # 生成标签
                    lab.append(str[k])
                lab = ''.join(lab)
                labn = lab

            if str[2] == "I":
                end = i+1
            i += 1
        else:
            labelsx.append(n)
            labelsx.append(end)
            labelsx.append(labn)         # 将每一个病历的最后一组bio传入labelsx

            textx = ''.join(text)
            labels = list_of_groups(labelsx,3)
            dicx = {"text":textx,"labels":labels}
            text = []
            labelsx = []
            i = 0
            json.dump(dicx, output, ensure_ascii=False)
            output.write("\n")

    f.close()
    output.close()


txtdir = "../assets/train.txt"
jsondir = "../assets/traint.json"
transform_txt_json(txtdir,jsondir)