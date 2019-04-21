#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time, sys
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import math

import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_g, SRGAN_d, Vgg19_simple_api
from utils import *
from config import config, log_config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
#import imageio.core.util
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse

import ocr
'''def silence_imageio_warning(*args, **kwargs):
    pass

imageio.core.util._precision_warn = silence_imageio_warning
'''
###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))
do_ocr = False
def convert(img):
    #print(img.dtype)
    img = np.multiply(np.add(img,1), 127.5).astype('uint8')
    return img

def quant(res, org):
    res = res[:,:,0]
    org = org[:,:,0]
    #print(img1)
    '''mse = np.mean( (img1 - img2) ** 2 )
    #print(img1.shape)
    if mse == 0:
        return 0, 100
    PIXEL_MAX = 255.0
    return mse, 20 * math.log10(PIXEL_MAX / math.sqrt(mse)), ss
    '''
    ss = (1+ssim(res, org, data_range=res.max() - res.min()))/2
    ms = mse(res, org)
    ps = psnr(res, org)
    return ms, ps, ss
    
def train():
    print("Training...")
    ## create folders to save result images and trained model
    print("Init epochs:", n_epoch_init, "\nAdversarial epochs:", n_epoch)
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    logfile = open("logfile.txt", "a")
    min_dim_hr = 32
    min_dim_lr = 8
    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))#[:21]
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))#[:21]
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))#[:21]
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))#[:21]
    
    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    for i in range(len(train_hr_imgs)):
        if len(train_hr_imgs[i].shape)==2:
            train_hr_imgs[i] = np.stack((train_hr_imgs[i],)*3, axis=-1)
        #print(train_hr_imgs[i].shape)
        min_dim_hr = min(min_dim_hr, min(train_hr_imgs[i].shape[0], train_hr_imgs[i].shape[1]))
    print("Images in dataset:", len(train_hr_imgs))
    #return
    min_dim_lr = min_dim_hr//4
            #size = valid_lr_img.shape
    # for im in train_hr_imgs:
    #     print(im.shape)
    # valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, min_dim_lr, min_dim_lr, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, min_dim_hr, min_dim_hr, 3], name='t_target_image')

    net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    net_g.print_params(False)
    net_g.print_layers()
    net_d.print_params(False)
    net_d.print_layers()

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(
        t_target_image, size=[224, 224], method=0,
        align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)  # resize_generate_image_for_vgg
    
    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)

    ## test inference
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    #g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    vgg_loss = 2e-3 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)
    #EPSILON = 1e-6
    #rho_loss = 1e-1 * tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(net_g.outputs, t_target_image)) + EPSILON))
    g_loss = mse_loss + vgg_loss + g_gan_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    ## SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    #tf.global_variables_initializer()
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        logfile.write("  Loading %s: %s, %s\n" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()
    '''
    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs = train_hr_imgs[0:batch_size]
    # sample_imgs = tl.vis.read_images(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print('sample HR sub-image:', sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_ginit + '/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_ginit + '/_train_sample_384.png')
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_gan + '/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_gan + '/_train_sample_384.png')
    '''
    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    logfile.write(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init + 1):
        #print("running epoch", epoch)
        #logfile.write("running epoch"+str(epoch)+"\n")
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        #print("Loading images batch")
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            ## update G
            #print("running session", n_iter)
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            #print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)
        logfile.write(log+'\n')
        
        ## quick evaluation on train set
        #if (epoch != 0) and (epoch % 10 == 0):
        #out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})  #; print('gen sub-image:', out.shape, out.min(), out.max())
        #print("[*] save images")
#            tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)

        ## save model
        #if (epoch != 0) and (epoch % 10 == 0):
        tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
        #print("G init model saved")
        logfile.write("G init model saved\n")
    ###========================= train GAN (SRGAN) =========================###
    print("\n\nStarted adversarial learning\n\n")
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch != 0 and epoch!= n_epoch and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)
        
        logfile.write(log+'\n')
        
        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0
        
        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        #print("Loading images batch")
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            ## update D
            errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            ## update G
            #errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            #print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" %
            #      (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
            total_d_loss += errD
            total_g_loss += errG
            #print("running session", n_iter)
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                                                                                total_g_loss / n_iter)
        print(log)
        logfile.write(log+'\n')
        #
        #logfile.write("Completed")
        #logfile.close()
        ## quick evaluation on train set
        #if (epoch != 0) and (epoch % 10 == 0):
        #out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})  #; print('gen sub-image:', out.shape, out.min(), out.max())
        #print("[*] save images")
        #tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)
        
        ## save model
        #if (epoch != 0) and (epoch % 10 == 0):
        
        tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
        tl.files.save_npz(net_d.all_params, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)
        if(epoch%25==0):
            conceval(epoch, net_g_test, t_image, sess)
        logfile.write("G and D models saved")
    logfile.write("Completed")
    logfile.close()

def conceval(epoch, net_g_test, t_image, sess):
    print("Intermediate Evaluating epoch...", epoch)
    interlog = open("inter_eval.txt", 'a')
    tot_psnr = 0
    tot_mse = 0
    tot_ssim = 0
    tot_res_acc = 0
    tot_hr_acc = 0
    tot_lr_acc = 0
    tot_bic_acc = 0
    res_beats_hr = 0
    res_beats_bic = 0
    res_beats_lr = 0
    res_fails = 0
    global do_ocr
    do_ocr = True
    test_set_size = 16
    test_outputs = []
    ## create folders to save result images
    save_dir = "samples/intermediate"
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"
    if(do_ocr):
        print("Evaluating with OCR")
    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))[:test_set_size]
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))[:test_set_size]
    #for i in valid_hr_img_list:
    #    print (i)
    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    num_lr_imgs = len(valid_lr_imgs)
    num_hr_imgs = len(valid_hr_imgs)
    '''print("loaded", num_lr_imgs, "LR images")
    if(mode=='multi' and num_lr_imgs != num_hr_imgs):
        print('Unequal images in LR and HR')
        return
    if(mode=='single' and (num_lr_imgs==0 or num_hr_imgs==0)):
        print('No images found')
        return
    sample_imgs = train_hr_imgs[0:batch_size]
    # sample_imgs = tl.vis.read_images(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print('sample HR sub-image:', sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    '''
    ###========================== RESTORE G =============================###
    in_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    net_g_oth = SRGAN_g(in_image, is_train=False, reuse=True)
    ses2 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tf.global_variables_initializer()
    tl.files.load_and_assign_npz(sess=ses2, name=checkpoint_dir + '/g_srgan.npz', network=net_g_oth)
    print("Loaded model\nProcessing images...")
    ###======================= EVALUATION =============================###
    for imid in range(num_lr_imgs): #64  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
        valid_lr_img = valid_lr_imgs[imid]
        valid_hr_img = valid_hr_imgs[imid]
        img_name = valid_lr_img_list[imid]
        #print("Processing image :\t", imid, "/\t", num_lr_imgs, "\t", img_name)
        # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
          # rescale to ［－1, 1]
        # print(valid_lr_img.min(), valid_lr_img.max())
        size = valid_lr_img.shape
        if len(size)==2:
            valid_lr_img = np.stack((valid_lr_img,)*3, axis=-1)
            valid_hr_img = np.stack((valid_hr_img,)*3, axis=-1)
            size = valid_lr_img.shape
        # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size
        #print("size", size)
        valid_lr_img_res = (valid_lr_img / 127.5) - 1
        start_time = time.time()
        out = ses2.run(net_g_oth.outputs, {in_image: [valid_lr_img_res]})
        #print("took: %4.4fs" % (time.time() - start_time))
        out_uint8 = convert(out[0])
        #print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        #print("[*] save images\n")
        test_outputs.append(out_uint8)
        tl.vis.save_image(out_uint8, save_dir + '/'+img_name[:-4]+'_gen_'+format(epoch, '03d')+'.png')
        #tl.vis.save_image(valid_lr_img, save_dir + '/'+img_name[:-4]+'_lr_'+format(epoch, '03d')+'.png')
        #tl.vis.save_image(valid_hr_img, save_dir + '/'+img_name[:-4]+'_hr_'+format(epoch, '03d')+'.png')
        out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
        #tl.vis.save_image(out_bicu, save_dir + '/'+img_name[:-4]+'_bicubic.png')
        #print(type(out_uint8), out_uint8.shape)
        #print(type(valid_hr_img), valid_hr_img.shape)
        #print(type(valid_lr_img), valid_lr_img.shape)
        img_mse, img_psnr, img_ssim = quant(out_uint8, valid_hr_img)
        #print('===')
        tot_psnr += img_psnr
        tot_mse += img_mse
        tot_ssim += img_ssim
        if(do_ocr):
            res_acc, hr_acc, lr_acc, bic_acc = ocr.getAccuracy(out_uint8, valid_hr_img, valid_lr_img, out_bicu, imid)
            tot_res_acc += res_acc
            tot_hr_acc += hr_acc
            tot_lr_acc += lr_acc
            tot_bic_acc += bic_acc
            if(res_acc>hr_acc):
                res_beats_hr += 1
            elif(res_acc>bic_acc):
                res_beats_bic +=1
            elif(res_acc>lr_acc):
                res_beats_lr += 1
            else:
                res_fails += 1
    if(do_ocr):
        ocrres = "Average GEN accuracy: "+str(tot_res_acc/num_lr_imgs)[:8]+"\nAverage HRI accuracy: "+str(tot_hr_acc/num_lr_imgs)[:8]+"\nAverage LRI accuracy: "+str(tot_lr_acc/num_lr_imgs)[:8]+"\nAverage BIC accuracy: "+str(tot_bic_acc/num_lr_imgs)[:8]+"\n\nRES>HRI: "+str(res_beats_hr)+"\nRES>BIC: "+str(res_beats_bic)+"\nRES>LRI: "+str(res_beats_lr)+"\nRESFAIL: "+str(res_fails)+'\n'+'='*50+'\n'
        #ocrlog.write(ocrres)
        try:
            hist = "Average PSNR: "+str(tot_psnr/num_lr_imgs)[:8]+"\tAverage MSE: "+str(tot_mse/num_lr_imgs)[:8]+"\tAverage SSIM: "+str(tot_ssim/num_lr_imgs)[:8]+"\tAverage Improvement over bicubic: "+str(tot_res_acc/tot_bic_acc)[:8]+'\n'
        except:
            hist = "Average PSNR: "+str(tot_psnr/num_lr_imgs)[:8]+"\tAverage MSE: "+str(tot_mse/num_lr_imgs)[:8]+"\tAverage SSIM: "+str(tot_ssim/num_lr_imgs)[:8]+'\n'
    else:
        hist = "Average PSNR: "+str(tot_psnr/num_lr_imgs)[:8]+"\tAverage MSE: "+str(tot_mse/num_lr_imgs)[:8]+"\tAverage SSIM: "+str(tot_ssim/num_lr_imgs)[:8]+'\n'
    interlog.write(hist)
    interlog.close()
    ses2.close()
    del in_image
    del net_g_oth
    del test_outputs
    print("\nAll images done\n"+hist)
    return
    
def evaluate(mode):
    print("Evaluating...")
    history = open("eval_history.txt", "a")
    latest = open("latest_eval.txt", "w")
    ocrlog = open("ocrlog.txt", "a")
    tot_psnr = 0
    tot_mse = 0
    tot_ssim = 0
    tot_res_acc = 0
    tot_hr_acc = 0
    tot_lr_acc = 0
    tot_bic_acc = 0
    res_beats_hr = 0
    res_beats_bic = 0
    res_beats_lr = 0
    res_fails = 0
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"
    if(do_ocr):
        print("Evaluating with OCR")
    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))
    
    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()
    
    ###========================== DEFINE MODEL ============================###
    num_lr_imgs = len(valid_lr_imgs)
    num_hr_imgs = len(valid_hr_imgs)
    print("loaded", num_lr_imgs, "LR images")
    if(mode=='multi' and num_lr_imgs != num_hr_imgs):
        print('Unequal images in LR and HR')
        return
    if(mode=='single' and (num_lr_imgs==0 or num_hr_imgs==0)):
        print('No images found')
        return
    
    ###========================== RESTORE G =============================###
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    net_g = SRGAN_g(t_image, is_train=False, reuse=False)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    #tf.global_variables_initializer()
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g)
    print("Loaded model\nProcessing images...")
    ###======================= EVALUATION =============================###
    for imid in range(num_lr_imgs): #64  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
        valid_lr_img = valid_lr_imgs[imid]
        valid_hr_img = valid_hr_imgs[imid]
        img_name = valid_lr_img_list[imid]
        #print("Processing image :\t", imid, "/\t", num_lr_imgs, "\t", img_name)
        # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
          # rescale to ［－1, 1]
        # print(valid_lr_img.min(), valid_lr_img.max())
        size = valid_lr_img.shape
        if len(size)==2:
            valid_lr_img = np.stack((valid_lr_img,)*3, axis=-1)
            valid_hr_img = np.stack((valid_hr_img,)*3, axis=-1)
            size = valid_lr_img.shape
        # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size
        #print("size", size)
        valid_lr_img_res = (valid_lr_img / 127.5) - 1
        start_time = time.time()
        out = sess.run(net_g.outputs, {t_image: [valid_lr_img_res]})
        #print("took: %4.4fs" % (time.time() - start_time))
        out_uint8 = convert(out[0])
        #print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        #print("[*] save images\n")
        tl.vis.save_image(out_uint8, save_dir + '/'+img_name[:-4]+'_gen.png')
        tl.vis.save_image(valid_lr_img, save_dir + '/'+img_name[:-4]+'_lr.png')
        tl.vis.save_image(valid_hr_img, save_dir + '/'+img_name[:-4]+'_hr.png')
        out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
        tl.vis.save_image(out_bicu, save_dir + '/'+img_name[:-4]+'_bicubic.png')
        #print(type(out[0]), out[0].shape)
        #print(type(valid_hr_img), valid_hr_img.shape)
        #print(type(valid_lr_img), valid_lr_img.shape)
        img_mse, img_psnr, img_ssim = quant(out_uint8, valid_hr_img)
        
        tot_psnr += img_psnr
        tot_mse += img_mse
        tot_ssim += img_ssim
        if(do_ocr):
            res_acc, hr_acc, lr_acc, bic_acc = ocr.getAccuracy(out_uint8, valid_hr_img, valid_lr_img, out_bicu, imid)
            tot_res_acc += res_acc
            tot_hr_acc += hr_acc
            tot_lr_acc += lr_acc
            tot_bic_acc += bic_acc
            if(res_acc>hr_acc):
                res_beats_hr += 1
            elif(res_acc>bic_acc):
                res_beats_bic +=1
            elif(res_acc>lr_acc):
                res_beats_lr += 1
            else:
                res_fails += 1
        eval_log = "Image: "+str(imid+1)+"\tPSNR: "+str(img_psnr)[:8]+"\tMSE: "+str(img_mse)[:8]+"\tSSIM: "+str(img_ssim)[:8]+'\n'
        latest.write(eval_log)
        #print(type(valid_lr_img), type(out_bicu))
        if(mode=='single'):
            num_lr_imgs = 1
            latest.close()
            history.close()
            print("\n1 image done\n"+eval_log)
            return
        incre = int(50.0 / num_lr_imgs * imid)
        sys.stdout.write('\r' + '|%s%s| %d/%d images done' % ('\033[7m' + ' '*incre + ' \033[27m',' '*(49-incre), imid+1, num_lr_imgs))
        sys.stdout.flush()
    if(do_ocr):
        ocrres = "Average GEN accuracy: "+str(tot_res_acc/num_lr_imgs)[:8]+"\nAverage HRI accuracy: "+str(tot_hr_acc/num_lr_imgs)[:8]+"\nAverage LRI accuracy: "+str(tot_lr_acc/num_lr_imgs)[:8]+"\nAverage BIC accuracy: "+str(tot_bic_acc/num_lr_imgs)[:8]+"\n\nRES>HRI: "+str(res_beats_hr)+"\nRES>BIC: "+str(res_beats_bic)+"\nRES>LRI: "+str(res_beats_lr)+"\nRESFAIL: "+str(res_fails)+'\n'+'='*50+'\n'
        ocrlog.write(ocrres)
        try:
            hist = "Average PSNR: "+str(tot_psnr/num_lr_imgs)[:8]+"\tAverage MSE: "+str(tot_mse/num_lr_imgs)[:8]+"\tAverage SSIM: "+str(tot_ssim/num_lr_imgs)[:8]+"\tAverage Improvement over bicubic: "+str(tot_res_acc/tot_bic_acc)[:8]+'\n'
        except:
            hist = "Average PSNR: "+str(tot_psnr/num_lr_imgs)[:8]+"\tAverage MSE: "+str(tot_mse/num_lr_imgs)[:8]+"\tAverage SSIM: "+str(tot_ssim/num_lr_imgs)[:8]+'\n'
    else:
        hist = "Average PSNR: "+str(tot_psnr/num_lr_imgs)[:8]+"\tAverage MSE: "+str(tot_mse/num_lr_imgs)[:8]+"\tAverage SSIM: "+str(tot_ssim/num_lr_imgs)[:8]+'\n'
    history.write(hist)
    latest.close()
    history.close()
    ocrlog.close()
    print("\nAll images done\n"+hist)
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    print("Starting...")
    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate, evaluateall')
    parser.add_argument('--ocr', type=str, default='false', help='true, false')
    args = parser.parse_args()
    ocr_do = args.ocr
    if ocr_do == 'true':
        do_ocr = True
    else:
        do_ocr = False
    tl.global_flag['mode'] = args.mode
    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate(mode='single')
    elif tl.global_flag['mode'] == 'evaluateall':
        evaluate(mode='multi')
    else:
        raise Exception("Unknow --mode")
