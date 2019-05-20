import utils
import tf_utils
from build_data import build_data
import numpy as np
import tensorflow as tf
import sys
import os.path
import os

'Train the model on the train set and evaluate on the evaluation and test sets until ' \
'(1) maximum epochs limit or (2) early stopping break'
def checkInputs():
    if (len(sys.argv) <= 3) or os.path.isfile(sys.argv[0])==False :
        raise ValueError(
            'The configuration file and the timestamp should be specified.')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"  

gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True  


if __name__ == "__main__":

    # checkInputs()

    config=build_data("./configs/ADE/bio_config")
    

    train_data = utils.HeadData(config.train_id_docs, np.arange(len(config.train_id_docs)))  ## build data 
    # dev_data = utils.HeadData(config.dev_id_docs, np.arange(len(config.dev_id_docs)))
    # test_data = utils.HeadData(config.test_id_docs, np.arange(len(config.test_id_docs)))
    train_data, dev_data = train_data.split(0.2)
    dev_data, test_data = dev_data.split(0.5)

    tf.reset_default_graph()
    tf.set_random_seed(1)

    utils.printParameters(config)

    with tf.Session(config=gpuConfig) as sess:
        embedding_matrix = tf.get_variable('embedding_matrix', shape=config.wordvectors.shape, dtype=tf.float32,
                                           trainable=False).assign(config.wordvectors)
        emb_mtx = sess.run(embedding_matrix)

        model = tf_utils.model(config, emb_mtx, sess)

        obj, m_op, transition_params1, entity1Scores, predEntity1, transition_params2, entity2Scores, predEntity2 = model.run()

        train_step = model.get_train_op(obj)

        operations=tf_utils.operations(train_step, obj, m_op, transition_params1, entity1Scores, predEntity1, transition_params2, entity2Scores, predEntity2)


        sess.run(tf.global_variables_initializer())

        best_score=0
        min_loss = 0
        nepoch_no_imprv = 0  # for early stopping

        best_test_score = 0

        # print(config.dataset_set_bio_tags)
        # print(config.dataset_set_bio_relation_ners)

        for iter in range(config.nepochs+1):

            loss = model.train(train_data, operations, iter)

            dev_score=model.evaluate(dev_data, operations,'dev')

            test_score = model.evaluate(test_data, operations,'test')

            if dev_score >= best_score:
                nepoch_no_imprv = 0
                best_score = dev_score

                print ("- Best dev score {} so far in {} epoch".format(dev_score, iter))
            if test_score >= best_test_score:
                nepoch_no_imprv = 0
                best_test_score = test_score

                print("- Best test score {} so far in {} epoch".format(test_score, iter))

            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= config.nepoch_no_imprv:

                    print ("- early stopping {} epochs without " \
                                     "improvement".format(nepoch_no_imprv))

                    with open("./es_best.txt", "w+") as myfile:
                        myfile.write(str(iter))
                        myfile.close()

                    break

        print ("- Best dev score {} so far".format(best_score))
        print ("- Best test score {} so far".format(best_test_score))


