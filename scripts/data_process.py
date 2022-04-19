from data_utils import *

def test():
    # 抽取已标注数据的标签情况
    b_doccano_dataset_label_view('train.json',['招标项目编号'],1)
    # 从未标注数据中选取数据
    db = b_extrct_data_from_db_basic('tender')
    df_db = pd.DataFrame(db)
    # 查看未标注数据的关键词情况
    b_doccano_cat_data(df_db,100,['招标编号','招标项目编号'])
    # 保存未标注数据
    b_save_df_datasets(df_db,'test2.json')
    # 模型标注数据
    b_label_dataset('test2.json')
    # 合并标注数据标签和原始数据

    # 将json训练数据转换为bio标签
    bio_labels = b_generate_biolabels_from('labels.txt')
    # 将json训练数据转换为bio训练数据 train_trf.json文件
    b_trans_dataset_bio(bio_labels,'train.json')
    b_trans_dataset_bio(bio_labels,'dev.json')
    # 划分数据集为更小的集合
    b_bio_split_dataset_by_max('train_trf.json',510)
    b_bio_split_dataset_by_max('dev_trf.json',510)





