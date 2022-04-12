import json
def transform_json_txt(jsondir,txtdir):
    """
    jsondir: json文件路径
    txtdir：转换后txt文件的路径
    """
    f = open(jsondir,'r',encoding='utf-8')
    ln = 0
    list = []
    str1 = 'B-'
    str2 = 'I-'
    str3 = 'O'
    for line in f.readlines():    # 读取文件中每一行
        ln += 1
        dic = json.loads(line)
        # id = dic['id']            # json中对象转换为Python中的字典
        text = dic['text']

        text = text.replace(' ','-')     # 将text中所有空格转换为‘-’，因为labels中数字将空格也对应上去了
        text = ' '.join(text)            # 将text中每个字符以空格隔开
        text = text.split()              # 以空格为分隔符对text进行切片

        labels = dic['label']
        for n in range(0, len(text)):
            if len(labels) == 0:
                list.append(text[n])
                list.append(str3 + "\n")
                text[n] = ' '.join(list)
                list = []
            else:
                for l in range(0, len(labels)):
                    if n == labels[l][0]:
                        list.append(text[n])
                        list.append(str1+labels[l][2]+"\n")
                        text[n] = ' '.join(list)
                        list = []
                        break
                    elif n > labels[l][0] and n < labels[l][1]:
                        list.append(text[n])
                        list.append(str2 + labels[l][2]+"\n")
                        text[n] = ' '.join(list)
                        list = []
                        break
                    elif l == len(labels)-1:
                        list.append(text[n])
                        list.append(str3+"\n")
                        text[n] = ' '.join(list)
                        list = []

        # print(text)
        f1 = open(txtdir, 'a', encoding='utf-8')
        f1.writelines(text)
        f1.write("\n")
    # f.write(str(ln))
    f.close()
    f1.close()

jsondir = "../assets/results.json"
txtdir = "../assets/results.txt"
transform_json_txt(jsondir, txtdir)