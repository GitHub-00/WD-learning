import tensorflow as tf
import sys
import os

def get_feature_column():
    '''
    age,workclass,fnlwgt,education,education-num,marital-status,occupation,
    relationship,race,sex,captial-gain,captial-loss,hours-per-week,native-country,income

    get wide feature and deep feature
    Return:
        wide feature column and deep feature column
    '''
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education-num')
    captial_gain = tf.feature_column.numeric_column('captial-gain')
    captial_loss = tf.feature_column.numeric_column('captial-loss')
    hours_per_week = tf.feature_column.numeric_column('hours-per-week')

    work_class = tf.feature_column.categorical_column_with_hash_bucket('workclass', hash_bucket_size=512)
    education = tf.feature_column.categorical_column_with_hash_bucket('education', hash_bucket_size=512)
    marital_status = tf.feature_column.categorical_column_with_hash_bucket('marital-status', hash_bucket_size=512)
    occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=512)
    relationship = tf.feature_column.categorical_column_with_hash_bucket('relationship', hash_bucket_size=512)

    age_bucket = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    gain_bucket = tf.feature_column.bucketized_column(captial_gain, boundaries=[0, 1000, 2000, 3000, 10000])
    loss_bucket = tf.feature_column.bucketized_column(captial_loss, boundaries=[0, 1000, 2000, 3000, 5000])

    cross_columns = [
        tf.feature_column.crossed_column([age_bucket, gain_bucket], hash_bucket_size=36),
        tf.feature_column.crossed_column([gain_bucket, loss_bucket], hash_bucket_size=16)
    ]

    base_column = [work_class, education, marital_status, occupation, relationship, age_bucket, gain_bucket, loss_bucket]
    wide_columns = base_column + cross_columns
    deep_columns = [
        age,
        education_num,
        captial_gain,
        captial_loss,
        hours_per_week,
        tf.feature_column.embedding_column(work_class, 9),
        tf.feature_column.embedding_column(education, 9),
        tf.feature_column.embedding_column(marital_status, 9),
        tf.feature_column.embedding_column(occupation, 9),
        tf.feature_column.embedding_column(relationship, 9),
    ]

    return wide_columns, deep_columns


def build_model_estimator(wide_column, deep_column, model_foder):
    '''
    Args:
        wide_column: wide feature
        deep_column: deep feature
        model_foder: origin model output folder
    Return:
        model_es, seving_input_fn
    '''
    model_es = tf.estimator.DNNLinearCombinedClassifier(
        model_dir = model_foder,
        linear_feature_columns = wide_column,
        linear_optimizer = tf.train.FtrlOptimizer(0.1,l2_regularization_strength=1.0),
        dnn_feature_columns = deep_column,
        dnn_optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=0.1,
                                                          l1_regularization_strength=0.001,
                                                          l2_regularization_strength=0.001),
        dnn_hidden_units = [128, 64, 32, 16]
    )
    feature_column = wide_column + deep_column
    feature_spec = tf.feature_column.make_parse_example_spec(feature_column)
    serving_input_fn = (tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
    return model_es, serving_input_fn

def input_fn(data_file, re_time, shuffle, batch_num, predict):
    '''
    Args:
        data_file: input_data, train_data, test_data
        re_time: to repeat the data file
        shuffle: shuffle or not
        batch_num:
        predict: train or test
    Retrun:
        train feature, train lable, or test feature
    '''
    _CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                            [0], [0], [0], [''], ['']]
    _CSV_COLUMNS = [
    'age','workclass','fnlwgt','education','education-num','marital-status','occupation',
    'relationship','race','sex','captial-gain','captial-loss','hours-per-week','native-country','income'
    ]

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income')
        classes = tf.equal(labels, ' >50K')
        return features, classes

    def parse_csv_predict(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income')
        return features

    data_set = tf.data.TextLineDataset(data_file).skip(1).filter(lambda line:tf.not_equal(tf.strings.regex_full_match(line,'.*\?.*'), True))
    if shuffle:
        data_set = data_set.shuffle(buffer_size=30000)
    if predict:
        data_set = data_set.map(parse_csv_predict, num_parallel_calls=5)
    else:
        data_set = data_set.map(parse_csv, num_parallel_calls=5)

    data_set = data_set.repeat(re_time)
    data_set = data_set.batch(batch_num)
    return data_set

def train_wd_model(model_es, train_file, test_file, model_export_folder, serving_input_fn):
    '''
    Args:
        model_es: wide and deep estimator
        train_file:
        test_file:marital-status
        model_export_folder: model export for tf serving
        serving_input_fn: function for model export
    '''
    total_run = 6
    #for index in range(total_run):
    model_es.train(input_fn=lambda:input_fn(train_file,20,True,100,False))
    print(model_es.evaluate(input_fn=lambda:input_fn(test_file,1,False,100,False)))
        #model_es.evaluate(input_fn=lambda: input_fn(test_file, 1, False, 100, False))
    model_es.export_savedmodel(model_export_folder, serving_input_fn)

def get_auc(predict_list, test_label):
    total_list = []
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        label = test_label[index]
        total_list.append((label, predict_score))
    sorted_total_list = sorted(total_list, key=lambda ele:ele[1])
    neg_num = 0
    pos_num = 0
    count = 1
    total_pos_index = 0
    for zuhe in sorted_total_list:
        label, predict_score = zuhe
        if label == 0:
            neg_num += 1
        else:
            pos_num += 1
            total_pos_index +=count
        count += 1

    auc_score = (total_pos_index - (pos_num)*(pos_num+1)/2)/(pos_num*neg_num)
    print('auc%.5f' %(auc_score))

def get_test_label(test_file):
    '''
    get label of test file
    '''
    if not os.path.exists(test_file):
        return []
    f = open(test_file)
    linenum = 0
    test_label_list = []
    for line in f:
        if linenum==0:
            linenum+=1
            continue
        if '?' in line.strip():
            continue

        item = line.strip().split(',')
        label_str = item[-1]
        if label_str==' >50K':
            test_label_list.append(1)
        elif label_str == ' <=50K':
            test_label_list.append(0)
        else:
            print('error')
    f.close()
    return test_label_list

def test_model_performance(model_es, test_file):
    '''
    test model auc in test data
    '''
    test_label = get_test_label(test_file)

    predict_list = []
    result = model_es.predict(input_fn = lambda :input_fn(test_file,1,False,100,True))
    for one_res in result:
        if 'probabilities' in one_res:
            predict_list.append(one_res['probabilities'][1])

    get_auc(predict_list, test_label)


def run_main(train_file, test_file, model_folder, model_export_folder):
    '''
    Args:
        train_file:
        test_file:
        model_folder: origin model folder to put train model
        model_export_folder: for tf serving
    '''

    wide_column, deep_column = get_feature_column()
    model_es, serving_input_fn = build_model_estimator(wide_column, deep_column, model_folder)

    train_wd_model(model_es, train_file, test_file, model_export_folder, serving_input_fn)
    test_model_performance(model_es, test_file)


if __name__=='__main__':
    run_main('../data/train.txt','../data/test.txt','../data/wd','../data/wd_export')