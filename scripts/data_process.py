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
    b_conbine_dataset_label('test2_label_1.json','招标项目编号')




