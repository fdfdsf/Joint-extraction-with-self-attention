import utils
import time
import eval
import numpy as np

class model:
    """Set of classes and methods for training the model and computing the ner and head selection loss"""


    def __init__(self, config, emb_mtx, sess):
        """"Initialize data"""
        self.config=config
        self.emb_mtx=emb_mtx
        self.sess=sess
        self.is_train = True

        '''Initialize variable'''
        import tensorflow as tf
        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None])
        # self.is_train = tf.placeholder(tf.int32)

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None])

        self.embedding_ids = tf.placeholder(tf.int32, [None, None])  # [ batch_size  *   max_sequence ] word embedding

        self.token_ids = tf.placeholder(tf.int32, [None, None])  # [ batch_size  *   max_sequence ]

        self.entity1_tags_ids = tf.placeholder(tf.int32, [None, None])

        self.entity2_tags_ids = tf.placeholder(tf.int32, [None, None])

        self.k1 = tf.placeholder(tf.int32, [None])
        self.k2 = tf.placeholder(tf.int32, [None])

        # scoring_matrix_gold = tf.placeholder(tf.float32, [None, None, None])  # [ batch_size  *   max_sequence]


        self.tokens = tf.placeholder(tf.string, [None, None])  # [ batch_size  *   max_sequence]
        self.entity1_tags = tf.placeholder(tf.string, [None, None])  # [ batch_size  *   max_sequence]
        self.entity2_tags = tf.placeholder(tf.string, [None, None])  # [ batch_size  *   max_sequence]

        # classes = ...
        self.seqlen = tf.placeholder(tf.int32, [None])  # [ batch_size ]

        self.doc_ids = tf.placeholder(tf.string, [None])  # [ batch_size ]


        self.dropout_embedding_keep = tf.placeholder(tf.float32, name="dropout_embedding_keep")
        self.dropout_lstm_keep = tf.placeholder(tf.float32, name="dropout_lstm_keep")
        self.dropout_lstm_output_keep = tf.placeholder(tf.float32, name="dropout_lstm_output_keep")
        self.dropout_fcl_ner_keep = tf.placeholder(tf.float32, name="dropout_fcl_ner_keep")

        self.embedding_matrix = tf.get_variable(name="embeddings", shape=self.emb_mtx.shape, initializer=tf.constant_initializer(self.emb_mtx), trainable=False)
        self.K = tf.get_variable(name="char_embeddings", dtype=tf.float32, shape=[len(self.config.dataset_set_characters), self.config.char_embeddings_size])



    def getEvaluator(self):
        if self.config.evaluation_method == "strict" and self.config.ner_classes == "BIO":  # the most common metric
            return eval.chunkEvaluator(self.config, ner_chunk_eval="boundaries_type", rel_chunk_eval="boundaries_type")
        elif self.config.evaluation_method == "boundaries" and self.config.ner_classes == "BIO":  # s
            return eval.chunkEvaluator(self.config, ner_chunk_eval="boundaries", rel_chunk_eval="boundaries")
        elif self.config.evaluation_method == "relaxed" and self.config.ner_classes == "EC":  # todo
            return eval.relaxedChunkEvaluator(self.config, rel_chunk_eval="boundaries_type")
        else:
            raise ValueError(
                'Valid evaluation methods : "strict" and "boundaries" in "BIO" mode and "relaxed" in "EC" mode .')


    def train(self, train_data, operations, iter):
        self.is_train = True

        loss = 0

        evaluator = self.getEvaluator()
        if self.config.ner_classes == "BIO":
            tagset = self.config.dataset_set_bio_tags
            tag2id = {k: v for v, k in enumerate(tagset)}
            id2tag = {v: k for v, k in enumerate(tagset)}

            relset = self.config.dataset_set_bio_relation_ners
            rel2id = {k: v for v, k in enumerate(relset)}
            id2rel = {v: k for v, k in enumerate(relset)}

        start_time = time.time()
        for x_train in utils.generator(train_data, operations.m_op, self.config, train=True):
            _, val, m_train, entity1Scores, predEntity1, entity2Scores, predEntity2 = self.sess.run(
                                            [operations.train_step, operations.obj, operations.m_op, 
                                            operations.entity1Scores, operations.predEntity1,
                                            operations.entity2Scores, operations.predEntity2], feed_dict=x_train)
            trueNer, trueRel, predNer, predRel = [], [], [], []
            for i in range(len(predEntity1)):
                trueSeq = m_train['entity1_tags_ids'][i]
                predSeq = predEntity1[i]
                # k1 = m_train['k1'][i]
                # k2 = m_train['k2'][i]
                tners1 = utils.ConvertToEntity(trueSeq, tag2id)
                pners1 = utils.ConvertToEntity(predSeq, tag2id)

                trueSeq2 = m_train['entity2_tags_ids'][i]
                predSeq2 = predEntity2[i]
                tners2, trels = utils.getRel(trueSeq2, rel2id, id2rel)
                pners2, prels = utils.getRel(predSeq2, rel2id, id2rel)

                trueNer, trueRel = utils.collectNerAndRel(tners1, tners2, trels)
                predNer, predRel = utils.collectNerAndRel(pners1, pners2, prels)

                evaluator.add(predNer, trueNer, predRel, trueRel)
            loss += val
            # break

        print('************iter %d****************' % (iter))
        print('----------Train----------')
        print('loss: %f ' % (loss))

        # if self.config.evaluation_method == "relaxed":
        #     evaluator.computeInfoMacro()
        # else:
        evaluator.printInfo()

        elapsed_time = time.time() - start_time
        print("Elapsed train time in sec:" + str(elapsed_time))
        return loss
        

    def evaluate(self, eval_data, operations, set):
        import tensorflow as tf
        self.is_train = False

        print('-------Evaluate on '+set+'-------')

        evaluator = self.getEvaluator()
        if self.config.ner_classes == "BIO":
            tagset = self.config.dataset_set_bio_tags
            tag2id = {k: v for v, k in enumerate(tagset)}
            id2tag = {v: k for v, k in enumerate(tagset)}

            relset = self.config.dataset_set_bio_relation_ners
            rel2id = {k: v for v, k in enumerate(relset)}
            id2rel = {v: k for v, k in enumerate(relset)}

        trueNers, trueRels = [], []
        predNers, predRels = [], []

        for x_dev, doc in utils.generator(eval_data, operations.m_op, self.config, train=False):
            
            predEntity1s = self.sess.run(operations.predEntity1, feed_dict=x_dev)

            for k in range(len(predEntity1s)):
                predNer, predRel = [], []
                # get Ture data
                trueNer, trueRel = utils.getNerAndRel(doc)

                # get predict data
                predEntity1 = predEntity1s[k]
                if self.config.ner1_loss == "crf":
                    i = 0
                # print(len(predEntity1))
                    while i < len(predEntity1):
                        tagid = predEntity1[i]
                        default = tag2id['O']
                        if tagid != default:
                            ner1 = utils.getNer(predEntity1, i, id2tag)
                            predNer.append(ner1)
                            end = ner1[2]
                            k1 = np.asarray([i])
                            k2 = np.asarray([end])
                            x_dev[operations.m_op['k1']] = np.asarray([i])
                            x_dev[operations.m_op['k2']] = np.asarray([end])
                            predEntity2s = self.sess.run(operations.predEntity2, feed_dict=x_dev)
                            predEntity2 = predEntity2s[0]
                            # print(len(predEntity2))
                            ners2, rels = utils.getRel(predEntity2, rel2id, id2rel)
                            predNer += ners2
                            assert len(ners2) == len(rels)
                            for j in range(len(rels)):
                                r = rels[j]
                                ner2 = ners2[j]
                                e1 = (ner1[1], ner1[2])
                                e2 = (ner2[1], ner2[2])
                                if e1 == e2:
                                    continue
                                predRel.append((e1, r, e2))
                            i = end + 1

                        else:
                            i += 1
                trueNers += trueNer
                trueRels += trueRel
                predNers += predNer
                predRels += predRel

        evaluator.add(predNers, trueNers, predRels, trueRels)
        evaluator.printInfo()
        return  evaluator.getChunkedOverallAvgF1()


    def get_train_op(self,obj):
        import tensorflow as tf

        if self.config.optimizer == 'Adam':

            optim = tf.train.AdamOptimizer(self.config.learning_rate)

        elif self.config.optimizer == 'Adagrad':
            optim = tf.train.AdagradOptimizer(self.config.learning_rate)
        elif self.config.optimizer == 'AdadeltaOptimizer':
            optim = tf.train.AdadeltaOptimizer(self.config.learning_rate)
        elif self.config.optimizer == 'GradientDescentOptimizer':
            optim = tf.train.GradientDescentOptimizer(self.config.learning_rate)

        if self.config.gradientClipping == True:

            gvs = optim.compute_gradients(obj)

            new_gvs = self.correctGradients(gvs)

            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in new_gvs]
            train_step = optim.apply_gradients(capped_gvs)


        else:
            train_step = optim.minimize(obj)

        return train_step

    def correctGradients(self,gvs):
        import tensorflow as tf

        new_gvs = []
        for grad, var in gvs:
            # print (grad)
            if grad == None:

                grad = tf.zeros_like(var)

            new_gvs.append((grad, var))
        if len(gvs) != len(new_gvs):
            print("gradient Error")
        return new_gvs


    def getNerScores1(self, lstm_out, n_types=1, dropout_keep_in_prob=1, reuse=False):
        import tensorflow as tf

        shape0 = lstm_out.get_shape()[-1]

        u_a = tf.get_variable("u_typ", [shape0, self.config.hidden_size_n1])  # [128 32] self.config.hidden_size_lstm * 2
        v = tf.get_variable("v_typ", [self.config.hidden_size_n1, n_types])  # [32,1] or [32,10]
        b_s = tf.get_variable("b_typ", [self.config.hidden_size_n1])
        b_c = tf.get_variable("b_ctyp", [n_types])

            # print(lstm_out.get_shape())
            # print(u_a.get_shape())

        mul = tf.einsum('aij,jk->aik', lstm_out, u_a)  # [16 348 64] * #[64 32] = [16 348 32]

        sum = mul + b_s
        if self.config.activation=="tanh":
            output = tf.nn.tanh(sum)
        elif self.config.activation=="relu":
            output = tf.nn.relu(sum)

        if self.config.use_dropout==True:
            output = tf.nn.dropout(output, keep_prob=dropout_keep_in_prob)

        g = tf.einsum('aik,kp->aip', output, v) + b_c
        return g
        

    def getNerScores2(self, lstm_out, n_types=1, dropout_keep_in_prob=1, reuse=False):
        import tensorflow as tf

        shape0 = lstm_out.get_shape()[-1]
        
        u_a = tf.get_variable("u_typ_", [shape0, self.config.hidden_size_n1])  # [384 32]  ## self.config.hidden_size_lstm * 6
        v = tf.get_variable("v_typ_", [self.config.hidden_size_n1, n_types])  # [32,1] or [32,10]
        b_s = tf.get_variable("b_typ_", [self.config.hidden_size_n1])
        b_c = tf.get_variable("b_ctyp_", [n_types])

            # print(lstm_out.get_shape())
            # print(u_a.get_shape())

        mul = tf.einsum('aij,jk->aik', lstm_out, u_a)  # [16 348 384] * #[384 32] = [16 348 32]

        sum = mul + b_s
        if self.config.activation=="tanh":
            output = tf.nn.tanh(sum)
        elif self.config.activation=="relu":
            output = tf.nn.relu(sum)

        if self.config.use_dropout==True:
            output = tf.nn.dropout(output, keep_prob=dropout_keep_in_prob)

        g = tf.einsum('aik,kp->aip', output, v) + b_c
        return g


    def getPosVec(self, input_rnn, pos):
        import tensorflow as tf
        '''
        input_rnn: [batch, seq_len, dim]
        pos: [batch, 1]
        return: [batch, dim]
        '''

        idxs = tf.range(0, tf.shape(input_rnn)[0])
        idxs = tf.expand_dims(idxs, 1)
        pos = tf.expand_dims(pos, 1)

        idxs = tf.concat([idxs, pos], 1)
        return tf.gather_nd(input_rnn, idxs)


    def getMidVec(self, input_rnn, pos1, pos2):
        import tensorflow as tf
        from tensorflow.python.ops import tensor_array_ops
        from tensorflow.python.ops import control_flow_ops

        batch_size = tf.shape(input_rnn)[0]
        ## seq to array
        data_ta = tensor_array_ops.TensorArray(
            dtype=input_rnn.dtype,
            size=batch_size,
            tensor_array_name='input_ta')
        data_ta = data_ta.unstack(input_rnn)
        ## pos1 and pos2 to array
        pos1_ta = tensor_array_ops.TensorArray(
            dtype=pos1.dtype,
            size=batch_size,
            tensor_array_name='pos1_ta')
        pos1_ta = pos1_ta.unstack(pos1)

        ## idx2 to array
        pos2_ta = tensor_array_ops.TensorArray(
            dtype=pos2.dtype,
            size=batch_size,
            tensor_array_name='pos2_ta')
        pos2_ta = pos2_ta.unstack(pos2)

        ## return
        return_ta = tensor_array_ops.TensorArray(
            dtype=input_rnn.dtype,
            size=batch_size,
            tensor_array_name='return_ta')

        loop = tf.constant(0, dtype='int32', name='loop')

        while_loop_kwargs = {
            'cond': lambda loop, *_: loop < batch_size,
            'parallel_iterations': 32,
            'swap_memory': True,
            'maximum_iterations': None}

        def _step(loop, output_ta_t):
            current_data = data_ta.read(loop) ## the loop-th data, the shape is [seq_len, s_size]
            current_pos1 = pos1_ta.read(loop) 
            current_pos2 = pos2_ta.read(loop) + 1
            output = tf.strided_slice(current_data, [current_pos1], [current_pos2])
            output = tf.reduce_sum(output, 0)

            output_ta_t = output_ta_t.write(loop, output)
            return (loop + 1, output_ta_t)

        final_outputs = control_flow_ops.while_loop(body=_step, loop_vars=(loop, return_ta), **while_loop_kwargs)

        last_loop = final_outputs[0]
        return_ta = final_outputs[1]
        outputs = return_ta.stack()

        return outputs


    def concatVec(self, input_rnn, vec):
        import tensorflow as tf
        '''
        input_rnn: [batch, seq_len, dim]
        vec: [batch, dim]
        return: [batch, seq_len, 2*dim]
        '''
        # print(vec.get_shape())
        # print(input_rnn[:, :, :1].get_shape())

        vec = tf.expand_dims(vec, 1)
        vec = tf.zeros_like(input_rnn[:, :, :1]) + vec
        return tf.concat([input_rnn, vec], axis=2)


    def normalize(self, inputs, epsilon = 1e-8, reuse=None):
        import tensorflow as tf
        with tf.variable_scope("nl", reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape),dtype=tf.float32)
            gamma = tf.Variable(tf.ones(params_shape),dtype=tf.float32)
            normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
            outputs = gamma * normalized + beta
        return outputs


    def self_attention(self, inputs, reuse=None):
        import tensorflow as tf
        with tf.variable_scope('multihead_attention', reuse=reuse):
            Q = tf.nn.relu(tf.layers.dense(inputs, self.config.attention_size * self.config.num_heads, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            K = tf.nn.relu(tf.layers.dense(inputs, self.config.attention_size * self.config.num_heads, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            V = tf.nn.relu(tf.layers.dense(inputs, self.config.attention_size * self.config.num_heads, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            Q_ = tf.concat(tf.split(Q, self.config.num_heads, axis=2), axis=0)
            K_ = tf.concat(tf.split(K, self.config.num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, self.config.num_heads, axis=2), axis=0)
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            key_masks = tf.sign(tf.abs(tf.reduce_sum(inputs, axis=-1)))
            key_masks = tf.tile(key_masks, [self.config.num_heads, 1])
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(inputs)[1], 1])
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
            outputs = tf.nn.softmax(outputs)
            query_masks = tf.sign(tf.abs(tf.reduce_sum(inputs, axis=-1)))
            query_masks = tf.tile(query_masks, [self.config.num_heads, 1])
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(inputs)[1]])
            outputs *= query_masks
            if self.is_train:
                outputs = tf.nn.dropout(outputs, keep_prob=self.dropout_lstm_output_keep)
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, self.config.num_heads, axis=0), axis=2)
            outputs += inputs
            outputs = self.normalize(outputs)
        return outputs


    def model1(self, lstm_output, reuse=False):
        import tensorflow as tf

        with tf.variable_scope("model1_computation", reuse=reuse):

            mask = tf.sequence_mask(self.seqlen, dtype=tf.float32)
            ## 1. model entity1 
            entity1_input = lstm_output
            # loss= tf.Print(loss, [tf.shape(loss)], 'shape of loss is:') # same as scoring matrix ie, [1 59 590]
            
            entity1Scores = self.getNerScores1(entity1_input, len(self.config.dataset_set_bio_tags), dropout_keep_in_prob=self.dropout_fcl_ner_keep, reuse=reuse)
            
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(entity1Scores, self.entity1_tags_ids, self.seqlen)
        
            if self.config.ner1_loss == "crf":

                lossEntity1 = -log_likelihood
                predEntity1, _ = tf.contrib.crf.crf_decode(entity1Scores, transition_params, self.seqlen)
                return lossEntity1, transition_params, entity1Scores, predEntity1

            elif self.config.ner1_loss == "softmax":
                lossEntity1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=entity1Scores, labels=self.entity1_tags_ids)

                predEntity1 = tf.cast(tf.arg_max(entity1Scores, 2), tf.int32)
                return lossEntity1, entity1Scores, predEntity1


    def model2(self, lstm_output, entity1Scores, predEntity1, reuse=False):
        import tensorflow as tf

        with tf.variable_scope("model2_computation", reuse=reuse):

            # 1. method1
            k1_vec = self.getPosVec(lstm_output, self.k1) ##  or lstm_output
            k2_vec = self.getPosVec(lstm_output, self.k2) ##  or lstm_output
            vec = tf.concat([k1_vec, k2_vec], axis=1)

            # 2. method2
            # vec = self.getMidVec(lstm_output, self.k1, self.k2)
            # lstm_output = tf.concat([lstm_output, entity1Scores], axis=2)

            entity2_input = self.concatVec(lstm_output, vec)

            if self.config.label_embeddings_size > 0:

                label_matrix = tf.get_variable(name="label_embeddings", dtype=tf.float32, shape=[len(self.config.dataset_set_bio_tags), self.config.label_embeddings_size])
                labels =  self.entity1_tags_ids if self.is_train  else predEntity1
                label_embeddings = tf.nn.embedding_lookup(label_matrix, labels)
                entity2_input = tf.concat([entity2_input, label_embeddings], axis=2)
                # rel_input = tf.concat([lstm_output, label_embeddings], axis=2)
            ## 2. model entity2

            entity2Scores = self.getNerScores2(entity2_input, len(self.config.dataset_set_bio_relation_ners), dropout_keep_in_prob=self.dropout_fcl_ner_keep, reuse=reuse)

            # entity2Scores = tf.Print(entity2Scores, [tf.shape(entity2_ids), entity2_ids, tf.shape(entity2Scores)], 'entity2_ids:  ', summarize=1000)
            
            log_likelihood2, transition_params2 = tf.contrib.crf.crf_log_likelihood(entity2Scores, self.entity2_tags_ids, self.seqlen)

            if self.config.ner2_loss == "crf":

                lossEntity2 = -log_likelihood2
                predEntity2, _ = tf.contrib.crf.crf_decode(entity2Scores, transition_params2, self.seqlen)

                return lossEntity2, transition_params2, entity2Scores, predEntity2

            elif self.config.ner2_loss == "softmax":
                lossEntity2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=entity2Scores, labels=self.entity2_tags_ids)
                predEntity2 = tf.cast(tf.arg_max(entity2Scores, 2), tf.int32)

                return lossEntity2, entity2Scores, predEntity2


    def lstm(self, reuse=False):
        import tensorflow as tf
        #####char embeddings

        # 1. get character embeddings
        with tf.variable_scope("lstm_computation", reuse=reuse):

            # K = tf.get_variable(name="char_embeddings", dtype=tf.float32,
            #                 shape=[len(self.config.dataset_set_characters), self.config.char_embeddings_size])
            # shape = (batch, sentence, word, dim of char embeddings)
            char_embeddings = tf.nn.embedding_lookup(self.K, self.char_ids)

        # 2. put the time dimension on axis=1 for dynamic_rnn
            s = tf.shape(char_embeddings)  # store old shape

            char_embeddings_reshaped = tf.reshape(char_embeddings, shape=[-1, s[-2], self.config.char_embeddings_size])
            word_lengths_reshaped = tf.reshape(self.word_lengths, shape=[-1])

            char_hidden_size = self.config.hidden_size_char

        # 3. bi lstm on chars
            cell_fw = tf.contrib.rnn.BasicLSTMCell(char_hidden_size, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(char_hidden_size, state_is_tuple=True)

            _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                                              inputs=char_embeddings_reshaped,
                                                                              sequence_length=word_lengths_reshaped,
                                                                              dtype=tf.float32)
        # shape = (batch x sentence, 2 x char_hidden_size)
            output = tf.concat([output_fw, output_bw], axis=-1)

        # shape = (batch, sentence, 2 x char_hidden_size)
            char_rep = tf.reshape(output, shape=[-1, s[1], 2 * char_hidden_size])

        # concat char embeddings

            word_embeddings = tf.nn.embedding_lookup(self.embedding_matrix, self.embedding_ids)

            if self.config.use_chars == True:
                input_rnn = tf.concat([word_embeddings, char_rep], axis=-1)

            else:
                input_rnn = word_embeddings

        # with tf.variable_scope("lstm_computation", reuse=False):

            if self.config.use_dropout:
                input_rnn = tf.nn.dropout(input_rnn, keep_prob=self.dropout_embedding_keep)
                    #input_rnn = tf.Print(input_rnn, [dropout_embedding_keep], 'embedding:  ', summarize=1000)
            for i in range(self.config.num_lstm_layers):
                if self.config.use_dropout and i > 0:
                    input_rnn = tf.nn.dropout(input_rnn, keep_prob=self.dropout_lstm_keep)
                    #input_rnn = tf.Print(input_rnn, [dropout_lstm_keep], 'lstm:  ', summarize=1000)

                lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size_lstm)
                # Backward direction cell
                lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size_lstm)

                lstm_out, _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=lstm_fw_cell,
                    cell_bw=lstm_bw_cell,
                    inputs=input_rnn,
                    sequence_length=self.seqlen,
                    dtype=tf.float32, scope='BiLSTM' + str(i))

                input_rnn = tf.concat(lstm_out, 2)

                lstm_output = input_rnn

            if self.config.attention_size > 0:
                lstm_output = self.self_attention(lstm_output, reuse=reuse)

            if self.config.use_dropout:
                lstm_output = tf.nn.dropout(lstm_output, keep_prob=self.dropout_lstm_output_keep)

            return lstm_output


    def run(self):
        import tensorflow as tf

        lstm_output = self.lstm(reuse=False)
        transition_params1 = None
        transition_params2 = None

        if self.config.ner1_loss == "crf":
            lossEntity1, transition_params1, entity1Scores, predEntity1 = self.model1(lstm_output, reuse=False)
            # lossEntity2, transition_params2, entity2Scores, predEntity2 = self.model2(lstm_output, entity1Scores, predEntity1, reuse=False)

        elif self.config.ner1_loss == "softmax":
            lossEntity1, entity1Scores, predEntity1 = self.model1(lstm_output, reuse=False)
            # lossEntity2, entity2Scores, predEntity2 = self.model2(lstm_output, predEntity1, reuse=False)

        if self.config.ner2_loss == "crf":
            # lossEntity1, transition_params1, entity1Scores, predEntity1 = self.model1(lstm_output, reuse=False)
            lossEntity2, transition_params2, entity2Scores, predEntity2 = self.model2(lstm_output, entity1Scores, predEntity1, reuse=False)

        elif self.config.ner2_loss == "softmax":
            # lossEntity1, entity1Scores, predEntity1 = self.model1(lstm_output, reuse=False)
            lossEntity2, entity2Scores, predEntity2 = self.model2(lstm_output, entity1Scores, predEntity1, reuse=False)

        obj = tf.reduce_sum(lossEntity1) + tf.reduce_sum(lossEntity2)
        #perturb the inputs
        raw_perturb = tf.gradients(obj, lstm_output)[0]  # [batch, L, dim]
        normalized_per = tf.nn.l2_normalize(raw_perturb, axis=[1, 2])
        perturb =self.config.alpha * tf.sqrt(tf.cast(tf.shape(lstm_output)[2], tf.float32)) * tf.stop_gradient(normalized_per)
        perturb_inputs = lstm_output + perturb

        if self.config.ner1_loss == "crf":
            lossEntity1_per, _, entity1Scores_per, predEntity1_per = self.model1(perturb_inputs, reuse=True)
            # lossEntity2_per, _, _, _ = self.model2(perturb_inputs, entity1Scores_per, predEntity1_per, reuse=True)

        elif self.config.ner1_loss == "softmax":
            lossEntity1_per, entity1Scores_per, predEntity1_per = self.model1(perturb_inputs, reuse=True)
            # lossEntity2_per, _, _ = self.model2(perturb_inputs, predEntity1_per, reuse=True)

        if self.config.ner2_loss == "crf":
            # lossEntity1_per, _, entity1Scores_per, predEntity1_per = self.model1(perturb_inputs, reuse=True)
            lossEntity2_per, _, _, _ = self.model2(perturb_inputs, entity1Scores_per, predEntity1_per, reuse=True)

        elif self.config.ner2_loss == "softmax":
            # lossEntity1_per, _, predEntity1_per = self.model1(perturb_inputs, reuse=True)
            lossEntity2_per, _, _ = self.model2(perturb_inputs, entity1Scores_per, predEntity1_per, reuse=True)

        # lossEntity1_per, _, label_matrix_per = self.model1(perturb_inputs, seqlen, dropout_fcl_ner_keep, entity1_tags_ids)
        # lossEntity2_per, _ = self.model2(perturb_inputs, label_matrix_per, seqlen, dropout_fcl_ner_keep, entity1_tags_ids, k1, k2, is_train, entity2_tags_ids)
            
        if self.config.use_adversarial==True:
            obj += tf.reduce_sum(lossEntity1_per) + tf.reduce_sum(lossEntity2_per)


        m = {}
        # m['isTrain'] = self.is_train
        m['embeddingIds'] = self.embedding_ids
        m['charIds'] = self.char_ids
        m['tokensLens'] = self.word_lengths
        m['entity1_tags_ids'] = self.entity1_tags_ids
        m['entity2_tags_ids'] = self.entity2_tags_ids
        m['seqlen'] = self.seqlen
        m['doc_ids'] = self.doc_ids
        m['tokenIds'] = self.token_ids
        m['dropout_embedding']=self.dropout_embedding_keep
        m['dropout_lstm']=self.dropout_lstm_keep
        m['dropout_lstm_output']=self.dropout_lstm_output_keep
        m['dropout_fcl_ner']=self.dropout_fcl_ner_keep
        m['tokens'] = self.tokens
        m['entity1_tags'] = self.entity1_tags
        m['entity2_tags'] = self.entity2_tags
        m['k1'] = self.k1
        m['k2'] = self.k2

        return obj, m, transition_params1, entity1Scores, predEntity1, transition_params2, entity2Scores, predEntity2


class operations():
    def __init__(self, train_step, obj, m_op, transition_params1, entity1Scores, predEntity1, transition_params2, entity2Scores, predEntity2):

        self.train_step=train_step
        self.obj=obj
        self.m_op = m_op
        self.transition_params1 = transition_params1
        self.transition_params2 = transition_params2
        self.entity1Scores = entity1Scores
        self.entity2Scores = entity2Scores
        self.predEntity1 = predEntity1
        self.predEntity2 = predEntity2
        