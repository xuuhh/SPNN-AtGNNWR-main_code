import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tensorflow.contrib import layers
import tensorflow.keras.layers
import data_import as data_import
import os
import shutil
import datetime
import math
import utils as utils
from data_import import DataSet
from SPNAtngnnwr import S_T_NETWORK
from SPNAtngnnwr import GTW_NETWORK
from SPNAtngnnwr import BASE_NETWORK
from SPNAtngnnwr import DIAGNOSIS
from tensorflow.python.client import timeline
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import io
from xlrd import open_workbook
from xlutils.copy import copy
from scipy.stats import t
import shutil


from affine import Affine
from functools import reduce

if __name__ == '__main__':

    # global setting
    diff_weight = False
    batch_norm = True
    create_force = True
    random_fixed = True
    base_path = '../dataset/'
    log_y = False
    seed = 10
    test_model = 1

    col_data_x = ['GDP', 'people', 'cz', 'ny','mt', 'cement']
    col_data_y = ['CO2']
    col_date = ['Monitor_Date']
    col_coordx = ['X']
    col_coordy = ['Y']

    if test_model == 0:
        models = ['gnnwr', 'gnnwr', 'gnnwr', 'gnnwr']
        datafile = 'wdataset.csv'
        log_y = False
        train_ratio = 0.7
        validation_ratio = 0.15
        st_weight_init = 'he'
        gtw_weigt_init = 'he'
        epochs = 2000
        start_lr = 0.1
        max_lr = 1.0
        total_up_steps = 20000
        up_decay_steps = 2000
        maintain_maxlr_steps = 20000
        delay_steps = 1000
        delay_rate = 0.98
        keep_prob_ratio = 0.8
        val_early_stopping = True
        val_early_stopping_begin_step = 5000
        model_comparison_criterion = 0.01

    else:
        models = ['dgnnwr', 'dgnnwr', 'dgnnwr', 'dgnnwr', 'dgnnwr']
        datafile = 'wdataset.csv'
        log_y = False
        train_ratio = 0.7
        validation_ratio = 0.15
        st_weight_init = 'he'
        gtw_weigt_init = 'he'
        epochs = 2000
        start_lr = 0.1
        max_lr = 1.0
        total_up_steps = 20000
        up_decay_steps = 2000
        maintain_maxlr_steps = 20000
        delay_steps = 1000
        delay_rate = 0.98
        keep_prob_ratio = 0.8
        val_early_stopping = True
        val_early_stopping_begin_step = 2000
        model_comparison_criterion = 0.01

    datafile_name = datafile[0:datafile.rfind('.')]

    for model_index in range(len(models)):
        if model_index > 0:
            tf.reset_default_graph()

        model = models[model_index % len(models)]

        if model == 'gnnwr':
            # 考虑空间，但不考虑空间network
            no_space = True
            s_no_network = True
            # 不考虑时间
            no_time = True
            t_no_network = True
            # 时空没有network
            st_no_network = True
            # 空间距离
            s_each_dir = False
            t_cycle = False
            # 数据相关设置
            data_path = base_path + datafile

            # 激活函数
            s_activate_fun = tf.identity
            t_activate_fun = tf.identity
            st_activate_fun = tf.identity
            gtw_activate_fun = "prelu"
        elif model == 'dgnnwr':
            # 考虑空间，但不考虑空间network
            no_space = True
            s_no_network = True
            # 不考虑时间
            no_time = True
            t_no_network = False
            # 时空没有network
            st_no_network = True
            # 空间距离
            s_each_dir = True
            t_cycle = False
            # 数据相关设置
            data_path = base_path + datafile

            # 激活函数
            s_activate_fun = tf.nn.relu
            t_activate_fun = tf.identity
            st_activate_fun = tf.identity
            gtw_activate_fun = "prelu"

        date_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")



        # reading data
        trainset, valiset, testset, miny, maxy, dataname = data_import.init_dataset(data_path, train_ratio=train_ratio,
                                                                                    validation_ratio=validation_ratio,
                                                                                    s_each_dir=s_each_dir,
                                                                                    t_cycle=t_cycle,
                                                                                    log_y=log_y, date_str=date_str,
                                                                                    create_force=create_force,
                                                                                    random_fixed=random_fixed,
                                                                                    seed=seed)


        # Training dataset
        x_train = trainset.x_data
        y_train = trainset.y_data
        s_dis_train = trainset.space_dis
        t_dis_train = trainset.time_dis
        # Validation dataset  验证数据集
        x_vali = valiset.x_data
        y_vali = valiset.y_data
        s_dis_vali = valiset.space_dis
        t_dis_vali = valiset.time_dis

        # Testing dataset
        x_test = testset.x_data
        y_test = testset.y_data
        s_dis_test = testset.space_dis
        t_dis_test = testset.time_dis

        sample_size = x_train.shape[0]
        # dropout的keep_prob
        keep_prob_st = tf.placeholder(tf.float32)
        keep_prob_gtw = tf.placeholder(tf.float32)
        bn_is_training = tf.placeholder(tf.bool)

        gtnnwr_output_size = trainset.x_data.shape[1]
        x_data = tf.placeholder(tf.float32, [None, gtnnwr_output_size])
        y_data = tf.placeholder(tf.float32, [None, 1])
        distance = tf.placeholder(tf.float32, [None, None])

        if not no_space:
            s_input_size = s_dis_train.shape[2]
            snn_hidden_layer_count = 1
            snn_neural_sizes = [3]
            snn_output_size = 1

            s_network = S_T_NETWORK('space', sample_size, s_input_size, snn_hidden_layer_count, snn_neural_sizes,
                                    s_activate_fun, keep_prob_st, output_layer_size=snn_output_size, batch_norm=batch_norm, bn_is_training=bn_is_training,
                                    weight_init=st_weight_init, diff_weight=diff_weight, no_network=s_no_network)

            snn_output_size = s_network.output_size
            snn_output = tf.reshape(s_network.output, [-1, sample_size, snn_output_size])
            snn_x_data = s_network.dist_data
        else:
            snn_output_size = 0
            snn_output = None
            snn_x_data = tf.placeholder(tf.float32, [None, None, None])

        if not no_time:
            t_input_size = t_dis_train.shape[2]
            tnn_hidden_layer_count = 1
            tnn_neural_sizes = [3]
            tnn_output_size = 2

            t_network = S_T_NETWORK('time', sample_size, t_input_size, tnn_hidden_layer_count, tnn_neural_sizes,
                                    t_activate_fun, keep_prob_st, output_layer_size=tnn_output_size, batch_norm=batch_norm, bn_is_training=bn_is_training,
                                    weight_init=st_weight_init, diff_weight=diff_weight, no_network=t_no_network)

            tnn_output_size = t_network.output_size
            tnn_output = tf.reshape(t_network.output, [-1, sample_size, tnn_output_size])
            tnn_x_data = t_network.dist_data
        else:
            tnn_output_size = 0
            tnn_output = None
            tnn_x_data = tf.placeholder(tf.float32, [None, None, None])

        # 对SNN和TNN的输入输出做转换，输入到STNN
        if no_space:
            st_input = tnn_output
        elif no_time:
            st_input = snn_output
        else:
            st_input = tf.concat([snn_output, tnn_output], 2)
        st_input_size = snn_output_size + tnn_output_size
        st_input = tf.reshape(st_input, [-1, st_input_size])

        stnn_hidden_layer_count = 0
        stnn_neural_sizes = [0]
        stnn_output_size = 1

        st_network = S_T_NETWORK('space_time', sample_size, st_input_size, stnn_hidden_layer_count, stnn_neural_sizes,
                                 st_activate_fun, keep_prob_st, output_layer_size=stnn_output_size, dist_data=st_input, batch_norm=batch_norm,
                                 bn_is_training=bn_is_training, weight_init=st_weight_init, diff_weight=diff_weight,
                                 no_network=st_no_network)

        gtwnn_hidden_layer_count, gtwnn_neural_sizes = utils.hidden_layers(sample_size, factor=4, hidden_node_limit=100, max_layer_count=3)

        stnn_output_size = st_network.output_size
        gtwnn_input = tf.reshape(st_network.output, [-1, sample_size * stnn_output_size])

        gtw_network = GTW_NETWORK('gtw_network', gtwnn_input, gtwnn_hidden_layer_count, gtwnn_neural_sizes, gtnnwr_output_size,
                                  gtw_activate_fun, keep_prob_gtw, batch_norm=True, bn_is_training=bn_is_training,
                                  weight_init=gtw_weigt_init)


        gtwnn_input = x_data

        gtwnn_hidden_layer_count, gtwnn_neural_sizes = utils.hidden_layers(sample_size, factor=4, hidden_node_limit=100,
                                                                           max_layer_count=3)

        gtw_network = GTW_NETWORK('gtw_network', gtwnn_input, gtwnn_hidden_layer_count, gtwnn_neural_sizes,
                                  gtnnwr_output_size,
                                  gtw_activate_fun, keep_prob_gtw, batch_norm=True, bn_is_training=bn_is_training,
                                  weight_init=gtw_weigt_init)

        # 多元线性回归
        linear_beta = tf.squeeze(utils.linear(tf.constant(trainset.x_data, dtype=tf.float32),
                                              tf.constant(np.reshape(trainset.y_data, [len(trainset.y_data), 1]),
                                                          dtype=tf.float32)))
        linear_beta_mat = tf.diag(linear_beta)
        # 回归部分
        yhat_rg = tf.diag_part(tf.matmul(tf.matmul(x_data, linear_beta_mat), tf.transpose(gtw_network.gtweight)))
        yhat = yhat_rg
        gtbeta = tf.transpose(tf.matmul(linear_beta_mat, tf.transpose(gtw_network.gtweight)))
        loss = tf.reduce_mean(tf.square(yhat - tf.squeeze(y_data)))

        #假设检验
        diagnosis = DIAGNOSIS(x_data, y_data, yhat, linear_beta_mat, gtw_network.gtweight, miny, maxy)
        loss_convert = diagnosis.loss_convert
        loss_add_reg = diagnosis.loss_add_reg
        ave_abs_error = diagnosis.ave_abs_error
        ave_rel_error = diagnosis.ave_rel_error

        r2_pearson = diagnosis.r2_pearson
        r2_coeff = diagnosis.r2_coeff
        r2_adjusted_coeff = diagnosis.r2_adjusted_coeff

        AICc = diagnosis.AICc
        AIC = diagnosis.AIC

        f1 = diagnosis.f1
        f2 = diagnosis.f2
        f3_dict = diagnosis.f3_dict
        f3_dict_2 = diagnosis.f3_dict_2

        # summary
        log_dir = 'Data/logs/' + datafile_name + '/' + model

        writer_train = tf.summary.FileWriter(log_dir + '/train/' + date_str, tf.get_default_graph())
        writer_validation = tf.summary.FileWriter(log_dir + '/validation/' + date_str, tf.get_default_graph())
        writer_test = tf.summary.FileWriter(log_dir + '/test/' + date_str, tf.get_default_graph())

        summary_loss = tf.summary.scalar('loss', loss)
        summary_pearson_r2 = tf.summary.scalar('r2_pearson', r2_pearson)
        summary_coeff_r2 = tf.summary.scalar('r2_coeff', r2_coeff)
        summary_AICc = tf.summary.scalar('AICc', AICc)
        summary_ave_abs_error = tf.summary.scalar('ave_abs_error', ave_abs_error)
        summary_ave_rel_error = tf.summary.scalar('ave_rel_error', ave_rel_error)
        summary_merged = tf.summary.merge_all()

        # 绘制结果
        plot_buf_ph = tf.placeholder(tf.string)
        image = tf.image.decode_png(plot_buf_ph, channels=4)
        image = tf.expand_dims(image, 0)  # make it batched
        #print(image.eval())
        plot_image_summary = tf.summary.image('result', image, max_outputs=10)

        global_step = tf.Variable(0, trainable=False)

        # 自定义学习率，先上升再下降
        learning_rate_decay = utils.exponential_decay_norm(start_lr, max_lr, global_step, delay_steps, delay_rate,
                                                           total_up_steps, up_decay_steps, maintain_maxlr_steps, staircase=True)

        # batch normalization 的初始化工作
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            train = tf.train.GradientDescentOptimizer(learning_rate_decay).minimize(loss_add_reg,
                                                                                    global_step=global_step)

        # 初始化变量
        init = tf.initialize_all_variables()

        feed_train = {x_data: x_train, y_data: y_train, keep_prob_gtw: 1, bn_is_training: True}
        feed_val = {x_data: x_vali, y_data: y_vali, keep_prob_gtw: 1, bn_is_training: False}
        feed_test = {x_data: x_test, y_data: y_test, keep_prob_gtw: 1, bn_is_training: False}

        #early stopping
        saver = tf.train.Saver()
        save_model_dir = 'Data/model_para/' + datafile_name + '/' + model + '/' + date_str + '/'
        if os.path.exists(save_model_dir):
            shutil.rmtree(save_model_dir)
        os.makedirs(save_model_dir)

        summary_step = 20
        early_stop_loss = loss
        val_early_stop_loss_min = 1000
        val_loss_noimprove_max_count = int(epochs / summary_step / 10)
        val_loss_no_improve_count = 0
        val_loss_best_step = 0

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        batch_size = int(trainset.num_examples / 10)
        batch_size = 2 ** math.floor(math.log2(batch_size))

        with tf.Session() as sess:
            sess.run(init)
            for step in range(epochs):
                batch = trainset.next_batch(batch_size)
                feed = {x_data: batch[0], y_data: batch[1], # snn_x_data: batch[2], tnn_x_data: batch[3],keep_prob_st: keep_prob_ratio,

                        keep_prob_gtw: keep_prob_ratio, bn_is_training: True}

                if step % delay_steps == 0:
                    cur_learning_rate = sess.run(learning_rate_decay)

                # 生成run_metadata
                if step % int(epochs / 5) == 0:
                    sess.run(train, feed_dict=feed, options=run_options,
                             run_metadata=run_metadata)
                    writer_train.add_run_metadata(run_metadata, 'step%d' % step)
                else:
                    sess.run(train, feed_dict=feed)

                if step % summary_step == 0:
                    # training summary
                    [summary, train_loss, train_r2_p, train_r2_c, train_r2_adjusted_c, yhat_train_temp,
                     cur_learning_rate] = sess.run(
                        [summary_merged, loss, r2_pearson, r2_coeff, r2_adjusted_coeff, yhat, learning_rate_decay],
                        feed_dict=feed_train)
                    writer_train.add_summary(summary, step)

                    # validation summary
                    [summary, val_loss, val_loss_add_reg, val_early_stop_loss, val_r2_p, val_r2_c,
                     yhat_val_temp] = sess.run(
                        [summary_merged, loss, loss_add_reg, early_stop_loss, r2_pearson, r2_coeff, yhat],
                        feed_dict=feed_val)
                    writer_validation.add_summary(summary, step)

                    # test summary
                    [summary] = sess.run(
                        [summary_merged], feed_dict=feed_test)
                    writer_test.add_summary(summary, step)

                    if val_early_stopping:
                        if step >= val_early_stopping_begin_step:
                            if val_early_stop_loss - val_early_stop_loss_min < 0:
                                val_early_stop_loss_min = val_early_stop_loss
                                model_file_name = os.path.join(save_model_dir, 'model')
                                saver.save(sess, model_file_name, global_step=step)  #保存模型
                                val_loss_no_improve_count = 0
                                val_loss_best_step = step
                            else:
                                val_loss_no_improve_count = val_loss_no_improve_count + 1
                                if val_loss_no_improve_count >= val_loss_noimprove_max_count:
                                    saver.restore(sess, saver.last_checkpoints[len(saver.last_checkpoints) - 1]) #恢复模型
                                    break

                    print('Training_' + str(step) + ' loss_rg: ' + str(train_loss) + ', r2_pearson: ' + str(
                        train_r2_p) + ', r2_coeff: ' + str(train_r2_c) + ', r2_adjusted_coeff: ' + str(
                        train_r2_adjusted_c) + ', cur_learning_rate: ' + str(cur_learning_rate) + '.')
                    print('Valdation_' + str(step) + ' loss_rg: ' + str(val_loss) + ', r2_pearson: ' + str(
                        val_r2_p) + ', r2_coeff: ' + str(val_r2_c) + ', early_stop_loss: ' + str(
                        val_early_stop_loss) + ', val_loss_best_step: ' + str(val_loss_best_step) + '.')

            print('Early Stopping Best Step: ' + str(val_loss_best_step))

            # output
            output_indictors = [diagnosis.RSS, diagnosis.MS_train, diagnosis.MS_common, r2_coeff, r2_adjusted_coeff,
                                diagnosis.A_R2, AICc, diagnosis.RMSE, ave_abs_error, ave_rel_error, diagnosis.r_pearson]
            output_indictor_names = ['RSS', 'MS_degree', 'MS_common', 'R2', 'Adjusted_R2_degree', 'Adjusted_R2', 'AICc',
                                     'RMSE', 'MAE', 'MAPE', 'r']

            # training final result
            [train_loss, yhat_train, train_R2_pearson, train_R2_coeff, gtweight_train, gtbeta_train,
             train_ave_abs_error, train_ave_rel_error, train_f1, train_f2, train_linear_beta, train_f3_dict] = sess.run(
                [loss, yhat, r2_pearson, r2_coeff, gtw_network.gtweight, gtbeta, ave_abs_error, ave_rel_error, f1, f2,
                 linear_beta, f3_dict], feed_dict=feed_train)
            print('Training dataset loss_rg: ' + str(train_loss) + ', r2_pearson: ' + str(
                train_R2_pearson) + ', r2_coeff: ' + str(train_R2_coeff) + '.')

            plot_buf = utils.get_plot_actual_pred_buf(y_train, yhat_train)
            image_summary = sess.run(plot_image_summary, feed_dict={plot_buf_ph: plot_buf.getvalue()})
            writer_train.add_summary(image_summary, global_step=0)

            plot_buf = utils.get_plot_actual_pred_buf(y_train, yhat_train, convert=True, miny=miny, maxy=maxy)
            image_summary = sess.run(plot_image_summary, feed_dict={plot_buf_ph: plot_buf.getvalue()})
            writer_train.add_summary(image_summary, global_step=1)

            # validation final result
            [val_loss, yhat_val, val_R2_pearson, val_R2_coeff, gtweight_val, gtbeta_val, val_ave_abs_error,
             val_ave_rel_error] = sess.run(
                [loss, yhat, r2_pearson, r2_coeff, gtw_network.gtweight, gtbeta, ave_abs_error, ave_rel_error],
                feed_dict=feed_val)
            print('Validation dataset loss_rg: ' + str(val_loss) + ', r2_pearson: ' + str(
                val_R2_pearson) + ', r2_coeff: ' + str(
                val_R2_coeff) + '.')

            plot_buf = utils.get_plot_actual_pred_buf(y_vali, yhat_val)
            image_summary = sess.run(plot_image_summary, feed_dict={plot_buf_ph: plot_buf.getvalue()})
            writer_validation.add_summary(image_summary, global_step=0)

            plot_buf = utils.get_plot_actual_pred_buf(y_vali, yhat_val, convert=True, miny=miny, maxy=maxy)
            image_summary = sess.run(plot_image_summary, feed_dict={plot_buf_ph: plot_buf.getvalue()})
            writer_validation.add_summary(image_summary, global_step=1)

            # test result
            [test_loss, yhat_test, test_R2_pearson, test_R2_coeff, gtweight_test, gtbeta_test, test_ave_abs_error,
             test_ave_rel_error] = sess.run(
                [loss, yhat, r2_pearson, r2_coeff, gtw_network.gtweight, gtbeta, ave_abs_error, ave_rel_error],
                feed_dict=feed_test)
            print('Testing dataset loss_rg: ' + str(test_loss) + ', r2_pearson: ' + str(
                test_R2_pearson) + ', r2_coeff: ' + str(
                test_R2_coeff) + '.')

            plot_buf = utils.get_plot_actual_pred_buf(y_test, yhat_test)
            image_summary = sess.run(plot_image_summary, feed_dict={plot_buf_ph: plot_buf.getvalue()})
            writer_test.add_summary(image_summary, global_step=0)

            plot_buf = utils.get_plot_actual_pred_buf(y_test, yhat_test, convert=True, miny=miny, maxy=maxy)
            image_summary = sess.run(plot_image_summary, feed_dict={plot_buf_ph: plot_buf.getvalue()})
            writer_test.add_summary(image_summary, global_step=1)


            # 结果输出
            output_indictors_train = sess.run(output_indictors, feed_dict=feed_train)
            output_indictors_val = sess.run(output_indictors, feed_dict=feed_val)
            output_indictors_test = sess.run(output_indictors, feed_dict=feed_test)

            # 随机后的原始数据存储路径
            file_save_path = 'Data/dataset/' + dataname + '.xls'
            # 结果输出路径
            result_save_path = 'Data/results/' + datafile_name + '/' + model + '/'
            if not os.path.exists(result_save_path):
                os.makedirs(result_save_path)
            result_file = result_save_path + dataname + '_' + date_str + '.xls'

            rb = open_workbook(file_save_path)
            wb = copy(rb)

            wb = utils.add_result_excel(wb, rb, 'train', maxy, miny, yhat_train, gtbeta_train, index=0)
            wb = utils.add_result_excel(wb, rb, 'validation', maxy, miny, yhat_val, gtbeta_val, index=len(yhat_train))
            wb = utils.add_result_excel(wb, rb, 'test', maxy, miny, yhat_test, gtbeta_test, index=len(yhat_train) + len(yhat_val))

            worksheet = wb.get_sheet('result')
            worksheet.write(0, 0, 'result')

            for i in range(len(output_indictor_names)):
                worksheet.write(0, 1 + i, output_indictor_names[i])
                worksheet.write(1, 1 + i, str(output_indictors_train[i]))
                worksheet.write(2, 1 + i, str(output_indictors_val[i]))
                worksheet.write(3, 1 + i, str(output_indictors_test[i]))

            curindex = len(output_indictor_names) + 1

            worksheet.write(0, curindex, 'f1')
            worksheet.write(0, curindex + 1, 'f2')
            worksheet.write(0, curindex + 2, 'f3_param1')
            worksheet.write(0, curindex + 3, 'f3_param2')
            worksheet.write(0, curindex + 4, 'f3_param3')
            worksheet.write(0, curindex + 5, 'f3_param4')
            worksheet.write(0, curindex + 6, 'f3_param5')
            worksheet.write(0, curindex + 7, 'f3_param6')


            worksheet.write(1, curindex, train_f1.item())
            worksheet.write(1, curindex + 1, train_f2.item())
            worksheet.write(1, curindex + 2, train_f3_dict['f3_param_0'].item())
            worksheet.write(1, curindex + 3, train_f3_dict['f3_param_1'].item())
            worksheet.write(1, curindex + 4, train_f3_dict['f3_param_2'].item())
            worksheet.write(1, curindex + 5, train_f3_dict['f3_param_3'].item())
            worksheet.write(1, curindex + 6, train_f3_dict['f3_param_4'].item())
            worksheet.write(1, curindex + 7, train_f3_dict['f3_param_5'].item())


            worksheet.write(1, 0, 'train')
            worksheet.write(2, 0, 'val')
            worksheet.write(3, 0, 'test')

            worksheet.write(5, 0, 'linear_GDP_weight')
            worksheet.write(5, 1, 'linear_people_weight')
            worksheet.write(5, 2, 'linear_cz_weight')



            worksheet.write(5, 3, 'linear_ny_weight')
            worksheet.write(5, 4, 'linear_mt_weight')

            worksheet.write(5, 5, 'linear_cement_weight')

            worksheet.write(5, 6, 'linear_constant_weight')

            worksheet.write(6, 0, train_linear_beta[0].item())
            worksheet.write(6, 1, train_linear_beta[1].item())
            worksheet.write(6, 2, train_linear_beta[2].item())



            worksheet.write(6, 3, train_linear_beta[3].item())  # 5
            worksheet.write(6, 4, train_linear_beta[4].item())  # 6
            worksheet.write(6, 5, train_linear_beta[5].item())

            worksheet.write(6, 6, train_linear_beta[6].item())


            if os.path.exists(result_file):
                os.remove(result_file)
            wb.save(result_file)