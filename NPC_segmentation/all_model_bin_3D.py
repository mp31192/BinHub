from keras.models import *
from all_layer_bin import *
from keras import initializers
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
import tensorflow as tf

def RealThreeDNet_multi_small_single(input_shape=(None, None,None, 1)):

    groups = 2

    input1 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input1')

    # 128*128*64
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input1)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L1')(x)
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(x)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_2_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L2')(x)
    pool1 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool1 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_GN')(pool1)
    pool1 = LeakyReLU(alpha=0.1, name='L1_p')(pool1)

    # 128*128*64
    conv2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(pool1)
    # print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv2_1_GN')(conv2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_1_GN')(conv2)
    x = LeakyReLU(alpha=0.1, name='L3')(x)
    conv2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    # print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    x = LeakyReLU(alpha=0.1, name='L4')(x)
    pool2 = Conv3D(32, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool2_GN')(pool2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool2 = LeakyReLU(alpha=0.1, name='L4_p')(pool2)
    # print("pool2 shape:", pool2.shape)

    # 128*128
    conv3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool2)
    # print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L5')(x)
    conv3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # print("conv3 shape:", conv3.shape)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_2_GN')(conv3)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L6')(x)
    pool3 = Conv3D(64, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool3_GN')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool3 = LeakyReLU(alpha=0.1, name='L6_p')(pool3)

    # 64*64
    conv4 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_1_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L13')(x)
    conv4 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_2_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L14')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = Conv3D(64, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool4_GN')(pool4)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool4 = LeakyReLU(alpha=0.1, name='L14_p')(pool4)

    # 8*8
    conv5 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L15')(x)
    conv5 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L16')(x)
    # drop5 = Dropout(0.5)(conv5)

    # 8*8
    up1 = Conv3DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    up1 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv1_1_GN')(up1)
    up1 = LeakyReLU(alpha=0.1, name='L15_u')(up1)
    merge6 = merge([conv4, up1], mode='concat', concat_axis=-1)
    conv6 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(merge6)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_1_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L17')(x)
    conv6 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_2_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L18')(x)

    # 16*16
    up2 = Conv3DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    up2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv2_1_GN')(up2)
    up2 = LeakyReLU(alpha=0.1, name='L18_u')(up2)
    merge7 = merge([conv3, up2], mode='concat', concat_axis=-1)
    conv7 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(merge7)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_1_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1, name='L19')(x)
    conv7 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_2_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = LeakyReLU(alpha=0.1, name='L20')(x)

    # 32*32
    up3 = Conv3DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    up3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv3_1_GN')(up3)
    up3 = LeakyReLU(alpha=0.1, name='L20_u')(up3)
    merge8 = merge([conv2, up3], mode='concat', concat_axis=-1)
    conv8 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(merge8)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_1_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L21')(x)
    conv8 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_2_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L22')(x)

    # 64*64
    up4 = Conv3DTranspose(8, kernel_size=3, strides=2, padding='same')(x)
    up4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv4_1_GN')(up4)
    up4 = LeakyReLU(alpha=0.1, name='L22_u')(up4)
    merge9 = merge([conv1, up4], mode='concat', concat_axis=-1)
    conv9 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(merge9)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_1_GN')(conv9)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = LeakyReLU(alpha=0.1, name='L23')(x)
    # conv9 = Conv3D(16, 3,  padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_2_GN')(conv9)
    # x = LeakyReLU(alpha=0.1,name='L24')(x)

    # 128*128
    conv10 = Conv3D(1, 1, kernel_initializer='RandomNormal', activation='sigmoid')(x)

    model = Model(input=[input1], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def RealThreeDNet_multi_small_laji(input_shape=(None, None,None, 1)):

    groups = 8

    input1 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input1')
    input2 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input2')

    # 128*128*64
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input1)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L1')(x)
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(x)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_2_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L2')(x)
    pool1 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool1 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_GN')(pool1)
    pool1 = LeakyReLU(alpha=0.1, name='L1_p')(pool1)

    conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input2)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L1_2_1')(x)
    conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_2_2_GN')(conv1_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L1_2_2')(x)
    pool1_2 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool1_2 = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_pool1_2_GN')(pool1_2)
    pool1_2 = LeakyReLU(alpha=0.1, name='L1_2_p')(pool1_2)

    # concat_conv12 = concatenate([conv1, conv1_2], axis=-1)

    concat_t1t2 = concatenate([pool1, pool1_2], axis=-1)

    # 128*128*64
    conv2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(concat_t1t2)
    # print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv2_1_GN')(conv2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_1_GN')(conv2)
    x = LeakyReLU(alpha=0.1, name='L3')(x)
    conv2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    # print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    x = LeakyReLU(alpha=0.1, name='L4')(x)
    pool2 = Conv3D(32, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool2_GN')(pool2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool2 = LeakyReLU(alpha=0.1, name='L4_p')(pool2)
    # print("pool2 shape:", pool2.shape)

    # 128*128
    conv3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool2)
    # print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L5')(x)
    conv3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # print("conv3 shape:", conv3.shape)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_2_GN')(conv3)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L6')(x)
    pool3 = Conv3D(64, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool3_GN')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool3 = LeakyReLU(alpha=0.1, name='L6_p')(pool3)

    # 64*64
    conv4 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_1_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L13')(x)
    conv4 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_2_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L14')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = Conv3D(64, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool4_GN')(pool4)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool4 = LeakyReLU(alpha=0.1, name='L14_p')(pool4)

    # 8*8
    conv5 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L15')(x)
    conv5 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L16')(x)
    # drop5 = Dropout(0.5)(conv5)

    # 8*8
    up1 = Conv3DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    up1 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv1_1_GN')(up1)
    up1 = LeakyReLU(alpha=0.1, name='L15_u')(up1)
    merge6 = merge([conv4, up1], mode='concat', concat_axis=-1)
    conv6 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(merge6)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_1_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L17')(x)
    conv6 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_2_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L18')(x)

    # 16*16
    up2 = Conv3DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    up2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv2_1_GN')(up2)
    up2 = LeakyReLU(alpha=0.1, name='L18_u')(up2)
    merge7 = merge([conv3, up2], mode='concat', concat_axis=-1)
    conv7 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(merge7)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_1_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1, name='L19')(x)
    conv7 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_2_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = LeakyReLU(alpha=0.1, name='L20')(x)

    # 32*32
    up3 = Conv3DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    up3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv3_1_GN')(up3)
    up3 = LeakyReLU(alpha=0.1, name='L20_u')(up3)
    merge8 = merge([conv2, up3], mode='concat', concat_axis=-1)
    conv8 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(merge8)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_1_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L21')(x)
    conv8 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_2_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L22')(x)

    # 64*64
    up4 = Conv3DTranspose(8, kernel_size=3, strides=2, padding='same')(x)
    up4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv4_1_GN')(up4)
    up4 = LeakyReLU(alpha=0.1, name='L22_u')(up4)
    merge9 = merge([conv1, conv1_2, up4], mode='concat', concat_axis=-1)
    conv9 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(merge9)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_1_GN')(conv9)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = LeakyReLU(alpha=0.1, name='L23')(x)
    # conv9 = Conv3D(16, 3,  padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_2_GN')(conv9)
    # x = LeakyReLU(alpha=0.1,name='L24')(x)

    # 128*128
    conv10 = Conv3D(1, 1, kernel_initializer='RandomNormal', activation='sigmoid')(x)

    model = Model(input=[input1, input2], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def RealThreeDNet_multi_small_laji_code_decode(input_shape=(None, None,None, 1)):

    groups = 8

    input1 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input1')
    input2 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input2')
    #
    # input1 = Input(shape=(256, 256, 64, 1), name='input1')
    # input2 = Input(shape=(256, 256, 64, 1), name='input2')

    # 128*128*64
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input1)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L1')(x)
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L2')(x)
    # conv1 = ConvOffset3D(8, name='deform_conv_3d_1_1')(x)
    pool1 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_GN')(pool1)
    pool1 = LeakyReLU(alpha=0.1, name='L1_p')(pool1)

    conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input2)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L1_2')(x)
    pool0 = Conv3D(16, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool0 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool0_GN')(pool0)
    pool0 = LeakyReLU(alpha=0.1,name='L0_p')(pool0)
    deconv1_decode = Conv3DTranspose(16, kernel_size=3, strides=2, padding='same')(pool0)
    conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(deconv1_decode)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_2_2_GN')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L1_2_2')(x)
    # conv1_2 = ConvOffset3D(8, name='deform_conv_3d_1_2')(x)
    pool1_2 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool1_2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_2_GN')(pool1_2)
    pool1_2 = LeakyReLU(alpha=0.1, name='L1_2_p')(pool1_2)

    concat_t1t2 = concatenate([pool1,pool1_2],axis=-1)


    # conv2 = ConvOffset3D(32, name='deform_conv_3d_2_1')(concat_t1t2)
    conv2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(concat_t1t2)
    # print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    x = LeakyReLU(alpha=0.1, name='L4')(x)
    pool2 = Conv3D(32, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool2_GN')(pool2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool2 = LeakyReLU(alpha=0.1, name='L4_p')(pool2)
    # print("pool2 shape:", pool2.shape)

    # 128*128
    conv3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool2)
    # print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L5')(x)
    conv3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # print("conv3 shape:", conv3.shape)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_2_GN')(conv3)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L6')(x)
    pool3 = Conv3D(64, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool3_GN')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool3 = LeakyReLU(alpha=0.1, name='L6_p')(pool3)

    # 64*64
    conv4 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_1_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L13')(x)
    conv4 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_2_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L14')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = Conv3D(64, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool4_GN')(pool4)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool4 = LeakyReLU(alpha=0.1, name='L14_p')(pool4)

    # 8*8
    conv5 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L15')(x)
    conv5 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L16')(x)
    # drop5 = Dropout(0.5)(conv5)

    # 8*8
    up1 = Conv3DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    up1 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv1_1_GN')(up1)
    up1 = LeakyReLU(alpha=0.1, name='L17_u')(up1)
    merge6 = merge([conv4, up1], mode='concat', concat_axis=-1)
    conv6 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(merge6)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_1_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L17')(x)
    conv6 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_2_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L18')(x)

    # 16*16
    up2 = Conv3DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    up2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv2_1_GN')(up2)
    up2 = LeakyReLU(alpha=0.1, name='L18_u')(up2)
    merge7 = merge([conv3, up2], mode='concat', concat_axis=-1)
    conv7 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(merge7)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_1_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1, name='L19')(x)
    conv7 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_2_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = LeakyReLU(alpha=0.1, name='L20')(x)

    # 32*32
    up3 = Conv3DTranspose(16, kernel_size=3, strides=2, padding='same')(x)
    up3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv3_1_GN')(up3)
    up3 = LeakyReLU(alpha=0.1, name='L20_u')(up3)
    merge8 = merge([conv2, up3], mode='concat', concat_axis=-1)
    conv8 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(merge8)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_1_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L21')(x)
    conv8 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_2_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L22')(x)

    # 64*64
    up4 = Conv3DTranspose(8, kernel_size=3, strides=2, padding='same')(x)
    up4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv4_1_GN')(up4)
    up4 = LeakyReLU(alpha=0.1, name='L22_u')(up4)
    merge9 = merge([conv1, conv1_2, up4], mode='concat', concat_axis=-1)
    conv9 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(merge9)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_1_GN')(conv9)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = LeakyReLU(alpha=0.1, name='L23')(x)
    # conv9 = Conv3D(16, 3,  padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_2_GN')(conv9)
    # x = LeakyReLU(alpha=0.1,name='L24')(x)

    # 128*128
    conv10 = Conv3D(1, 1, kernel_initializer='RandomNormal', activation='sigmoid')(x)

    model = Model(input=[input1, input2], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def RealThreeDNet_multi_small_laji_code_decode_capsules(input_shape=(None, None,None, 1)):

    groups = 8

    input1 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input1')
    input2 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input2')

    # 128*128*64
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input1)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L1')(x)
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(x)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_2_GN')(conv1)
    x1 = LeakyReLU(alpha=0.1, name='L2')(x)
    pool1 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool1 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_GN')(pool1)
    pool1 = LeakyReLU(alpha=0.1, name='L1_p')(pool1)

    conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input2)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L1_2')(x)
    pool0 = Conv3D(16, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool0 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool0_GN')(pool0)
    pool0 = LeakyReLU(alpha=0.1,name='L0_p')(pool0)
    deconv1_decode = Conv3DTranspose(16, kernel_size=3, strides=2, padding='same')(pool0)
    # print("conv1 shape:", conv1.shape)
    conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(deconv1_decode)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_2_2_GN')(conv1_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    x2 = LeakyReLU(alpha=0.1, name='L1_2_2')(x)
    pool1_2 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool1_2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_2_GN')(pool1_2)
    pool1_2 = LeakyReLU(alpha=0.1, name='L1_2_p')(pool1_2)

    # concat_conv12 = concatenate([conv1,conv1_2],axis=-1)

    # conv_out_1 = Conv3D(1, 1, strides=1, padding='same', kernel_initializer='he_normal',activation='sigmoid')(concat_conv12)

    concat_t1t2 = concatenate([pool1,pool1_2],axis=-1)

    B, H, W, D, C = concat_t1t2.get_shape() # _, 512, 512, 16
    concat_t1t2_reshaped = layers.Reshape((H.value, W.value, D.value, 1, C.value))(concat_t1t2)

    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = Conv3DCapsuleLayer(kernel_size=3, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                    routings=2, name='primarycaps1')(concat_t1t2_reshaped)

    primary_caps = Conv3DCapsuleLayer(kernel_size=3, num_capsule=1, num_atoms=16, strides=1, padding='same',
                                    routings=1, name='primarycaps2')(primary_caps)

    concat_t1t2_parimary = layers.Reshape((H.value, W.value, D.value, C.value))(primary_caps)

    up_out = Conv3DTranspose(16, kernel_size=3, strides=2, padding='same')(concat_t1t2_parimary)
    conv_p_out = Conv3D(1, 1, padding='same', kernel_initializer='he_normal',activation='sigmoid')(up_out)

    # 128*128*64
    # pool1 = Conv3D(32, 3, strides=2, padding='same', kernel_initializer='he_normal')(concat_t1t2_parimary)
    # pool1 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_GN')(pool1)
    # pool1 = LeakyReLU(alpha=0.1, name='L1_p')(pool1)
    conv2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(concat_t1t2_parimary)
    # print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv2_1_GN')(conv2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_1_GN')(conv2)
    x = LeakyReLU(alpha=0.1, name='L3')(x)
    conv2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    # print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    x = LeakyReLU(alpha=0.1, name='L4')(x)
    pool2 = Conv3D(32, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool2_GN')(pool2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool2 = LeakyReLU(alpha=0.1, name='L4_p')(pool2)
    # print("pool2 shape:", pool2.shape)

    # 128*128
    conv3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool2)
    # print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L5')(x)
    conv3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # print("conv3 shape:", conv3.shape)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_2_GN')(conv3)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L6')(x)
    pool3 = Conv3D(64, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool3_GN')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool3 = LeakyReLU(alpha=0.1, name='L6_p')(pool3)

    # 64*64
    conv4 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_1_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L13')(x)
    conv4 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_2_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L14')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = Conv3D(64, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool4_GN')(pool4)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool4 = LeakyReLU(alpha=0.1, name='L14_p')(pool4)

    # 8*8
    conv5 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L15')(x)
    conv5 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L16')(x)
    # drop5 = Dropout(0.5)(conv5)

    # 8*8
    up1 = Conv3DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    up1 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv1_1_GN')(up1)
    up1 = LeakyReLU(alpha=0.1, name='L15_u')(up1)
    merge6 = merge([conv4, up1], mode='concat', concat_axis=-1)
    conv6 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(merge6)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_1_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L17')(x)
    conv6 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_2_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L18')(x)

    # 16*16
    up2 = Conv3DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    up2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv2_1_GN')(up2)
    up2 = LeakyReLU(alpha=0.1, name='L18_u')(up2)
    merge7 = merge([conv3, up2], mode='concat', concat_axis=-1)
    conv7 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(merge7)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_1_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1, name='L19')(x)
    conv7 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_2_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = LeakyReLU(alpha=0.1, name='L20')(x)

    # 32*32
    up3 = Conv3DTranspose(16, kernel_size=3, strides=2, padding='same')(x)
    up3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv3_1_GN')(up3)
    up3 = LeakyReLU(alpha=0.1, name='L20_u')(up3)
    merge8 = merge([conv2, up3], mode='concat', concat_axis=-1)
    conv8 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(merge8)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_1_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L21')(x)
    conv8 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_2_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L22')(x)

    # 64*64
    up4 = Conv3DTranspose(8, kernel_size=3, strides=2, padding='same')(x)
    up4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv4_1_GN')(up4)
    up4 = LeakyReLU(alpha=0.1, name='L22_u')(up4)
    merge9 = merge([conv1, conv1_2, up4], mode='concat', concat_axis=-1)
    conv9 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(merge9)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_1_GN')(conv9)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = LeakyReLU(alpha=0.1, name='L23')(x)
    # conv9 = Conv3D(16, 3,  padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_2_GN')(conv9)
    # x = LeakyReLU(alpha=0.1,name='L24')(x)

    # 128*128
    conv10 = Conv3D(1, 1, kernel_initializer='RandomNormal', activation='sigmoid')(x)

    model = Model(input=[input1, input2], output=[conv10,conv_p_out])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def RealThreeDNet_multi_small_laji_code_decode_deform_dilate(input_shape=(None, None,None, 1)):

    groups = 8

    # input1 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input1')
    # input2 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input2')

    input1 = Input(shape=(256, 256, 64, 1), name='input1')
    input2 = Input(shape=(256, 256, 64, 1), name='input2')

    # 128*128*64
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input1)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L1')(x)
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L2')(x)
    conv1 = ConvOffset3D(9, name='deform_conv_3d_1_1')(x)
    pool1 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_GN')(pool1)
    pool1 = LeakyReLU(alpha=0.1, name='L1_p')(pool1)

    conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input2)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L1_2')(x)
    pool0 = Conv3D(16, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool0 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool0_GN')(pool0)
    pool0 = LeakyReLU(alpha=0.1,name='L0_p')(pool0)
    deconv1_decode = Conv3DTranspose(16, kernel_size=3, strides=2, padding='same')(pool0)
    conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(deconv1_decode)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_2_2_GN')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L1_2_2')(x)
    conv1_2 = ConvOffset3D(9, name='deform_conv_3d_1_2')(x)
    pool1_2 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(conv1_2)
    pool1_2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_2_GN')(pool1_2)
    pool1_2 = LeakyReLU(alpha=0.1, name='L1_2_p')(pool1_2)

    concat_t1t2 = concatenate([pool1,pool1_2],axis=-1)


    conv2 = ConvOffset3D(18, name='deform_conv_3d_2_1')(concat_t1t2)
    conv2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(conv2)
    # print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    x = LeakyReLU(alpha=0.1, name='L4')(x)
    pool2 = Conv3D(32, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool2_GN')(pool2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool2 = LeakyReLU(alpha=0.1, name='L4_p')(pool2)
    # print("pool2 shape:", pool2.shape)

    # 128*128
    conv3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool2)
    # print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L5')(x)
    conv3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # print("conv3 shape:", conv3.shape)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_2_GN')(conv3)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L6')(x)
    pool3 = Conv3D(64, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool3_GN')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool3 = LeakyReLU(alpha=0.1, name='L6_p')(pool3)

    # 64*64
    conv4 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_1_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L13')(x)
    conv4 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_2_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L14')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = Conv3D(64, 3, dilation_rate=2,strides=1, padding='same', kernel_initializer='he_normal')(x)
    pool4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool4_GN')(pool4)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool4 = LeakyReLU(alpha=0.1, name='L14_p')(pool4)

    # 8*8
    conv5 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L15')(x)
    conv5 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L16')(x)
    # drop5 = Dropout(0.5)(conv5)

    # 8*8
    merge6 = merge([conv4, x], mode='concat', concat_axis=-1)
    conv6 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(merge6)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_1_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L17')(x)
    conv6 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_2_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L18')(x)

    # 16*16
    up2 = Conv3DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    up2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv2_1_GN')(up2)
    up2 = LeakyReLU(alpha=0.1, name='L18_u')(up2)
    merge7 = merge([conv3, up2], mode='concat', concat_axis=-1)
    conv7 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(merge7)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_1_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1, name='L19')(x)
    conv7 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_2_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = LeakyReLU(alpha=0.1, name='L20')(x)

    # 32*32
    up3 = Conv3DTranspose(16, kernel_size=3, strides=2, padding='same')(x)
    up3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv3_1_GN')(up3)
    up3 = LeakyReLU(alpha=0.1, name='L20_u')(up3)
    merge8 = merge([conv2, up3], mode='concat', concat_axis=-1)
    conv8 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(merge8)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_1_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L21')(x)
    conv8 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_2_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L22')(x)

    # 64*64
    up4 = Conv3DTranspose(8, kernel_size=3, strides=2, padding='same')(x)
    up4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv4_1_GN')(up4)
    up4 = LeakyReLU(alpha=0.1, name='L22_u')(up4)
    merge9 = merge([conv1, conv1_2, up4], mode='concat', concat_axis=-1)
    conv9 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(merge9)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_1_GN')(conv9)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = LeakyReLU(alpha=0.1, name='L23')(x)
    # conv9 = Conv3D(16, 3,  padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_2_GN')(conv9)
    # x = LeakyReLU(alpha=0.1,name='L24')(x)

    # 128*128
    conv10 = Conv3D(1, 1, kernel_initializer='RandomNormal', activation='sigmoid')(x)

    model = Model(input=[input1, input2], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def RealThreeDNet_multi_small_laji_code_decode_dilate(input_shape=(None, None,None, 1)):

    groups = 8

    # input1 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input1')
    # input2 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input2')

    input1 = Input(shape=(256, 256, 64, 1), name='input1')
    input2 = Input(shape=(256, 256, 64, 1), name='input2')

    # 128*128*64
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input1)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L1')(x)
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L2')(x)
    # conv1 = ConvOffset3D(8, name='deform_conv_3d_1_1')(x)
    pool1 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool1 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_GN')(pool1)
    pool1 = LeakyReLU(alpha=0.1, name='L1_p')(pool1)

    conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input2)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L1_2')(x)
    pool0 = Conv3D(16, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool0 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool0_GN')(pool0)
    pool0 = LeakyReLU(alpha=0.1,name='L0_p')(pool0)
    deconv1_decode = Conv3DTranspose(16, kernel_size=3, strides=2, padding='same')(pool0)
    conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(deconv1_decode)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_2_2_GN')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L1_2_2')(x)
    # conv1_2 = ConvOffset3D(8, name='deform_conv_3d_1_2')(x)
    pool1_2 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool1_2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_2_GN')(pool1_2)
    pool1_2 = LeakyReLU(alpha=0.1, name='L1_2_p')(pool1_2)

    concat_t1t2 = concatenate([pool1,pool1_2],axis=-1)


    # conv2 = ConvOffset3D(32, name='deform_conv_3d_2_1')(concat_t1t2)
    conv2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(concat_t1t2)
    # print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    x = LeakyReLU(alpha=0.1, name='L4')(x)
    pool2 = Conv3D(32, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool2_GN')(pool2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool2 = LeakyReLU(alpha=0.1, name='L4_p')(pool2)
    # print("pool2 shape:", pool2.shape)

    # 128*128
    conv3 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool2)
    # print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L5')(x)
    conv3 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    # print("conv3 shape:", conv3.shape)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_2_GN')(conv3)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L6')(x)
    pool3 = Conv3D(64, 3, dilation_rate=2, strides=1, padding='same', kernel_initializer='he_normal')(x)
    pool3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool3_GN')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool3 = LeakyReLU(alpha=0.1, name='L6_p')(pool3)

    # 64*64
    conv4 = Conv3D(128, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_1_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L13')(x)
    conv4 = Conv3D(128, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_2_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L14')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = Conv3D(128, 3, dilation_rate=2,strides=1, padding='same', kernel_initializer='he_normal')(x)
    pool4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool4_GN')(pool4)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool4 = LeakyReLU(alpha=0.1, name='L14_p')(pool4)

    # 8*8
    conv5 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L15')(x)
    conv5 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L16')(x)
    # drop5 = Dropout(0.5)(conv5)

    # 8*8
    merge6 = merge([conv4, x], mode='concat', concat_axis=-1)
    conv6 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(merge6)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_1_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L17')(x)
    conv6 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_2_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L18')(x)

    # 16*16
    # up2 = Conv3DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    # up2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv2_1_GN')(up2)
    # up2 = LeakyReLU(alpha=0.1, name='L18_u')(up2)
    merge7 = merge([conv3, x], mode='concat', concat_axis=-1)
    conv7 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(merge7)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_1_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1, name='L19')(x)
    conv7 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_2_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = LeakyReLU(alpha=0.1, name='L20')(x)

    # 32*32
    up3 = Conv3DTranspose(16, kernel_size=3, strides=2, padding='same')(x)
    up3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv3_1_GN')(up3)
    up3 = LeakyReLU(alpha=0.1, name='L20_u')(up3)
    merge8 = merge([conv2, up3], mode='concat', concat_axis=-1)
    conv8 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(merge8)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_1_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L21')(x)
    conv8 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_2_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L22')(x)

    # 64*64
    up4 = Conv3DTranspose(8, kernel_size=3, strides=2, padding='same')(x)
    up4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv4_1_GN')(up4)
    up4 = LeakyReLU(alpha=0.1, name='L22_u')(up4)
    merge9 = merge([conv1, conv1_2, up4], mode='concat', concat_axis=-1)
    conv9 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(merge9)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_1_GN')(conv9)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = LeakyReLU(alpha=0.1, name='L23')(x)
    # conv9 = Conv3D(16, 3,  padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_2_GN')(conv9)
    # x = LeakyReLU(alpha=0.1,name='L24')(x)

    # 128*128
    conv10 = Conv3D(1, 1, kernel_initializer='RandomNormal', activation='sigmoid')(x)

    model = Model(input=[input1, input2], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def RealThreeDNet_multi_small_laji_code_decode_dilate_inception(input_shape=(None, None,None, 1)):

    groups = 8

    # input1 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input1')
    # input2 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input2')

    input1 = Input(shape=(256, 256, 64, 1), name='input1')
    input2 = Input(shape=(256, 256, 64, 1), name='input2')

    # 128*128*64
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input1)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L1')(x)
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L2')(x)
    # conv1 = ConvOffset3D(8, name='deform_conv_3d_1_1')(x)
    pool1 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool1 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_GN')(pool1)
    pool1 = LeakyReLU(alpha=0.1, name='L1_p')(pool1)

    conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input2)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L1_2')(x)
    pool0 = Conv3D(16, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool0 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool0_GN')(pool0)
    pool0 = LeakyReLU(alpha=0.1,name='L0_p')(pool0)
    deconv1_decode = Conv3DTranspose(16, kernel_size=3, strides=2, padding='same')(pool0)
    conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(deconv1_decode)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_2_2_GN')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L1_2_2')(x)
    # conv1_2 = ConvOffset3D(8, name='deform_conv_3d_1_2')(x)
    pool1_2 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool1_2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_2_GN')(pool1_2)
    pool1_2 = LeakyReLU(alpha=0.1, name='L1_2_p')(pool1_2)

    concat_t1t2 = concatenate([pool1,pool1_2],axis=-1)


    # conv2 = ConvOffset3D(32, name='deform_conv_3d_2_1')(concat_t1t2)
    conv2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(concat_t1t2)
    # print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    x = LeakyReLU(alpha=0.1, name='L4')(x)
    pool2 = Conv3D(32, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool2_GN')(pool2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool2 = LeakyReLU(alpha=0.1, name='L4_p')(pool2)
    # print("pool2 shape:", pool2.shape)

    # 128*128
    conv3 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool2)
    # print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L5')(x)
    conv3 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    # print("conv3 shape:", conv3.shape)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_2_GN')(conv3)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L6')(x)
    pool3 = Conv3D(64, 3, dilation_rate=2, strides=1, padding='same', kernel_initializer='he_normal')(x)
    pool3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool3_GN')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool3 = LeakyReLU(alpha=0.1, name='L6_p')(pool3)

    # 64*64
    conv4 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_1_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L13')(x)
    conv4 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_2_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L14')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = Conv3D(64, 3, dilation_rate=2,strides=1, padding='same', kernel_initializer='he_normal')(x)
    pool4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool4_GN')(pool4)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool4 = LeakyReLU(alpha=0.1, name='L14_p')(pool4)

    # 8*8
    conv5 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L15')(x)
    conv5 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    conv5 = LeakyReLU(alpha=0.1, name='L16')(x)
    # drop5 = Dropout(0.5)(conv5)

    # 128*128
    conv3_2 = Conv3D(64, 3, dilation_rate=1, padding='same', kernel_initializer='he_normal')(pool2)
    # print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_2_1_GN')(conv3_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L5_2')(x)
    conv3_2 = Conv3D(64, 3, dilation_rate=1, padding='same', kernel_initializer='he_normal')(x)
    # print("conv3 shape:", conv3.shape)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_2_GN')(conv3)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_2_2_GN')(conv3_2)
    x = LeakyReLU(alpha=0.1, name='L6_2')(x)
    pool3_2 = Conv3D(64, 3, dilation_rate=1, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool3_2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool3_2_GN')(pool3_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool3_2 = LeakyReLU(alpha=0.1, name='L6_p_2')(pool3_2)

    # 64*64
    conv4_2 = Conv3D(64, 3, dilation_rate=1, padding='same', kernel_initializer='he_normal')(pool3_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_1_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_2_1_GN')(conv4_2)
    x = LeakyReLU(alpha=0.1, name='L13_2')(x)
    conv4_2 = Conv3D(64, 3, dilation_rate=1, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_2_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_2_2_GN')(conv4_2)
    x = LeakyReLU(alpha=0.1, name='L14_2')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4_2 = Conv3D(64, 3, dilation_rate=1,strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool4_2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool4_2_GN')(pool4_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool4_2 = LeakyReLU(alpha=0.1, name='L14_p_2')(pool4_2)

    # 8*8
    conv5_2 = Conv3D(64, 3, dilation_rate=1, padding='same', kernel_initializer='he_normal')(pool4_2)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_1_GN')(conv5_2)
    x = LeakyReLU(alpha=0.1, name='L15_2')(x)
    conv5_2 = Conv3D(64, 3, dilation_rate=1, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_2_GN')(conv5_2)
    conv5_2 = LeakyReLU(alpha=0.1, name='L16_2')(x)

    # 16*16
    up1_2 = Conv3DTranspose(64, kernel_size=3, strides=2, padding='same')(conv5_2)
    up1_2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv1_1_2_GN')(up1_2)
    up1_2 = LeakyReLU(alpha=0.1, name='L18_u_2')(up1_2)
    merge7_2 = merge([conv4_2, up1_2], mode='concat', concat_axis=-1)
    conv7_2 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(merge7_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_1_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_1_GN')(conv7_2)
    x = LeakyReLU(alpha=0.1, name='L19_2')(x)
    conv7_2 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_2_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_2_GN')(conv7_2)
    conv7_2 = LeakyReLU(alpha=0.1, name='L20_2')(x)

    # 16*16
    up2_2 = Conv3DTranspose(32, kernel_size=3, strides=2, padding='same')(conv7_2)
    up2_2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv2_1_2_GN')(up2_2)
    up2_2 = LeakyReLU(alpha=0.1, name='L18_u_3')(up2_2)
    merge8_2 = merge([conv3_2, up2_2], mode='concat', concat_axis=-1)
    conv8_2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(merge8_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_1_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_1_GN')(conv8_2)
    x = LeakyReLU(alpha=0.1, name='L19_3')(x)
    conv8_2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_2_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_2_GN')(conv8_2)
    conv8_2 = LeakyReLU(alpha=0.1, name='L20_3')(x)

    # 8*8
    merge6 = merge([conv4, conv5], mode='concat', concat_axis=-1)
    conv6 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(merge6)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_1_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L17')(x)
    conv6 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_2_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L18')(x)

    # 16*16
    # up2 = Conv3DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    # up2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv2_1_GN')(up2)
    # up2 = LeakyReLU(alpha=0.1, name='L18_u')(up2)
    merge7 = merge([conv3, x], mode='concat', concat_axis=-1)
    conv7 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(merge7)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_1_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1, name='L19')(x)
    conv7 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_2_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    conv7 = LeakyReLU(alpha=0.1, name='L20')(x)

    concat_inception = concatenate([conv7,conv8_2],axis=-1)

    # 32*32
    up3 = Conv3DTranspose(16, kernel_size=3, strides=2, padding='same')(concat_inception)
    up3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv3_1_GN')(up3)
    up3 = LeakyReLU(alpha=0.1, name='L20_u')(up3)
    merge8 = merge([conv2, up3], mode='concat', concat_axis=-1)
    conv8 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(merge8)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_1_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L21')(x)
    conv8 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_2_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L22')(x)

    # 64*64
    up4 = Conv3DTranspose(8, kernel_size=3, strides=2, padding='same')(x)
    up4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv4_1_GN')(up4)
    up4 = LeakyReLU(alpha=0.1, name='L22_u')(up4)
    merge9 = merge([conv1, conv1_2, up4], mode='concat', concat_axis=-1)
    conv9 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(merge9)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_1_GN')(conv9)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = LeakyReLU(alpha=0.1, name='L23')(x)
    # conv9 = Conv3D(16, 3,  padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_2_GN')(conv9)
    # x = LeakyReLU(alpha=0.1,name='L24')(x)

    # 128*128
    conv10 = Conv3D(1, 1, kernel_initializer='RandomNormal', activation='sigmoid')(x)

    model = Model(input=[input1, input2], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def RealThreeDNet_multi_small_laji_code_decode_dilate_inception_register(input_shape=(None, None,None, 1)):

    groups = 8

    # input1 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input1')
    # input2 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input2')

    input1 = Input(shape=(256, 256, 64, 1), name='input1')
    input2 = Input(shape=(256, 256, 64, 1), name='input2')

    # 128*128*64
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input1)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L1')(x)
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L2')(x)
    # conv1 = ConvOffset3D(8, name='deform_conv_3d_1_1')(x)
    pool1 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool1 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_GN')(pool1)
    pool1 = LeakyReLU(alpha=0.1, name='L1_p')(pool1)

    conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input2)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L1_2')(x)
    conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_2_2_GN')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L1_2_2')(x)
    # conv1_2 = ConvOffset3D(8, name='deform_conv_3d_1_2')(x)
    pool1_2 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool1_2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_2_GN')(pool1_2)
    pool1_2 = LeakyReLU(alpha=0.1, name='L1_2_p')(pool1_2)

    concat_t1t2 = concatenate([pool1,pool1_2],axis=-1)


    # conv2 = ConvOffset3D(32, name='deform_conv_3d_2_1')(concat_t1t2)
    conv2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(concat_t1t2)
    # print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    x = LeakyReLU(alpha=0.1, name='L4')(x)
    pool2 = Conv3D(32, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool2_GN')(pool2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool2 = LeakyReLU(alpha=0.1, name='L4_p')(pool2)
    # print("pool2 shape:", pool2.shape)

    # 128*128
    conv3 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool2)
    # print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L5')(x)
    conv3 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    # print("conv3 shape:", conv3.shape)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_2_GN')(conv3)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L6')(x)
    pool3 = Conv3D(64, 3, dilation_rate=2, strides=1, padding='same', kernel_initializer='he_normal')(x)
    pool3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool3_GN')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool3 = LeakyReLU(alpha=0.1, name='L6_p')(pool3)

    # 64*64
    conv4 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_1_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L13')(x)
    conv4 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_2_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L14')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = Conv3D(64, 3, dilation_rate=2,strides=1, padding='same', kernel_initializer='he_normal')(x)
    pool4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool4_GN')(pool4)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool4 = LeakyReLU(alpha=0.1, name='L14_p')(pool4)

    # 8*8
    conv5 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L15')(x)
    conv5 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    conv5 = LeakyReLU(alpha=0.1, name='L16')(x)
    # drop5 = Dropout(0.5)(conv5)

    # 128*128
    conv3_2 = Conv3D(64, 3, dilation_rate=1, padding='same', kernel_initializer='he_normal')(pool2)
    # print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_2_1_GN')(conv3_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L5_2')(x)
    conv3_2 = Conv3D(64, 3, dilation_rate=1, padding='same', kernel_initializer='he_normal')(x)
    # print("conv3 shape:", conv3.shape)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_2_GN')(conv3)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_2_2_GN')(conv3_2)
    x = LeakyReLU(alpha=0.1, name='L6_2')(x)
    pool3_2 = Conv3D(64, 3, dilation_rate=1, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool3_2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool3_2_GN')(pool3_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool3_2 = LeakyReLU(alpha=0.1, name='L6_p_2')(pool3_2)

    # 64*64
    conv4_2 = Conv3D(64, 3, dilation_rate=1, padding='same', kernel_initializer='he_normal')(pool3_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_1_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_2_1_GN')(conv4_2)
    x = LeakyReLU(alpha=0.1, name='L13_2')(x)
    conv4_2 = Conv3D(64, 3, dilation_rate=1, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_2_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_2_2_GN')(conv4_2)
    x = LeakyReLU(alpha=0.1, name='L14_2')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4_2 = Conv3D(64, 3, dilation_rate=1,strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool4_2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool4_2_GN')(pool4_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool4_2 = LeakyReLU(alpha=0.1, name='L14_p_2')(pool4_2)

    # 8*8
    conv5_2 = Conv3D(64, 3, dilation_rate=1, padding='same', kernel_initializer='he_normal')(pool4_2)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_1_GN')(conv5_2)
    x = LeakyReLU(alpha=0.1, name='L15_2')(x)
    conv5_2 = Conv3D(64, 3, dilation_rate=1, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_2_GN')(conv5_2)
    conv5_2 = LeakyReLU(alpha=0.1, name='L16_2')(x)

    # 16*16
    up1_2 = Conv3DTranspose(64, kernel_size=3, strides=2, padding='same')(conv5_2)
    up1_2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv1_1_2_GN')(up1_2)
    up1_2 = LeakyReLU(alpha=0.1, name='L18_u_2')(up1_2)
    merge7_2 = merge([conv4_2, up1_2], mode='concat', concat_axis=-1)
    conv7_2 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(merge7_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_1_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_1_GN')(conv7_2)
    x = LeakyReLU(alpha=0.1, name='L19_2')(x)
    conv7_2 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_2_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_2_GN')(conv7_2)
    conv7_2 = LeakyReLU(alpha=0.1, name='L20_2')(x)

    # 16*16
    up2_2 = Conv3DTranspose(32, kernel_size=3, strides=2, padding='same')(conv7_2)
    up2_2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv2_1_2_GN')(up2_2)
    up2_2 = LeakyReLU(alpha=0.1, name='L18_u_3')(up2_2)
    merge8_2 = merge([conv3_2, up2_2], mode='concat', concat_axis=-1)
    conv8_2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(merge8_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_1_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_1_GN')(conv8_2)
    x = LeakyReLU(alpha=0.1, name='L19_3')(x)
    conv8_2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_2_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_2_GN')(conv8_2)
    conv8_2 = LeakyReLU(alpha=0.1, name='L20_3')(x)

    # 8*8
    merge6 = merge([conv4, conv5], mode='concat', concat_axis=-1)
    conv6 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(merge6)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_1_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L17')(x)
    conv6 = Conv3D(64, 3, dilation_rate=2, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_2_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L18')(x)

    # 16*16
    # up2 = Conv3DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    # up2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv2_1_GN')(up2)
    # up2 = LeakyReLU(alpha=0.1, name='L18_u')(up2)
    merge7 = merge([conv3, x], mode='concat', concat_axis=-1)
    conv7 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(merge7)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_1_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1, name='L19')(x)
    conv7 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_2_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    conv7 = LeakyReLU(alpha=0.1, name='L20')(x)

    concat_inception = concatenate([conv7,conv8_2],axis=-1)

    # 32*32
    up3 = Conv3DTranspose(16, kernel_size=3, strides=2, padding='same')(concat_inception)
    up3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv3_1_GN')(up3)
    up3 = LeakyReLU(alpha=0.1, name='L20_u')(up3)
    merge8 = merge([conv2, up3], mode='concat', concat_axis=-1)
    conv8 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(merge8)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_1_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L21')(x)
    conv8 = Conv3D(16, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_2_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L22')(x)

    # 64*64
    up4 = Conv3DTranspose(8, kernel_size=3, strides=2, padding='same')(x)
    up4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv4_1_GN')(up4)
    up4 = LeakyReLU(alpha=0.1, name='L22_u')(up4)
    merge9 = merge([conv1, conv1_2, up4], mode='concat', concat_axis=-1)
    conv9 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(merge9)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_1_GN')(conv9)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x = LeakyReLU(alpha=0.1, name='L23')(x)
    # conv9 = Conv3D(16, 3,  padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_2_GN')(conv9)
    # x = LeakyReLU(alpha=0.1,name='L24')(x)

    # 128*128
    conv10 = Conv3D(1, 1, kernel_initializer='RandomNormal', activation='sigmoid')(x)

    model = Model(input=[input1, input2], output=[conv10])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model

def RealThreeDNet_multi_small_laji_code_decode_multiout(input_shape=(None, None,None, 1)):

    groups = 8

    input1 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input1')
    input2 = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1), name='input2')

    # 128*128*64
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input1)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L1')(x)
    conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(x)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_2_GN')(conv1)
    x = LeakyReLU(alpha=0.1, name='L2')(x)
    pool1 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool1 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_GN')(pool1)
    pool1 = LeakyReLU(alpha=0.1, name='L1_p')(pool1)

    conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input2)
    # print("conv1 shape:", conv1.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L1_2')(x)
    pool0 = Conv3D(16, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool0 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool0_GN')(pool0)
    pool0 = LeakyReLU(alpha=0.1,name='L0_p')(pool0)
    deconv1_decode = Conv3DTranspose(16, kernel_size=3, strides=2, padding='same')(pool0)
    # print("conv1 shape:", conv1.shape)
    conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(deconv1_decode)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv1_2_2_GN')(conv1_2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_2_GN')(conv1_2)
    x = LeakyReLU(alpha=0.1, name='L1_2_2')(x)
    pool1_2 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool1_2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool1_2_GN')(pool1_2)
    pool1_2 = LeakyReLU(alpha=0.1, name='L1_2_p')(pool1_2)

    # concat_conv12 = concatenate([conv1,conv1_2],axis=-1)

    # conv_out_1 = Conv3D(1, 1, strides=1, padding='same', kernel_initializer='he_normal',activation='sigmoid')(concat_conv12)

    concat_t1t2 = concatenate([pool1,pool1_2],axis=-1)

    # 128*128*64
    conv2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(concat_t1t2)
    # print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv2_1_GN')(conv2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_1_GN')(conv2)
    x = LeakyReLU(alpha=0.1, name='L3')(x)
    conv2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    # print("conv2 shape:", conv2.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    x = LeakyReLU(alpha=0.1, name='L4')(x)
    pool2 = Conv3D(32, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool2_GN')(pool2)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool2 = LeakyReLU(alpha=0.1, name='L4_p')(pool2)
    # print("pool2 shape:", pool2.shape)

    # 128*128
    conv3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool2)
    # print("conv3 shape:", conv3.shape)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_1_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L5')(x)
    conv3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # print("conv3 shape:", conv3.shape)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv3_2_GN')(conv3)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
    x = LeakyReLU(alpha=0.1, name='L6')(x)
    pool3 = Conv3D(64, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool3_GN')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool3 = LeakyReLU(alpha=0.1, name='L6_p')(pool3)

    # 64*64
    conv4 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool3)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_1_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L13')(x)
    conv4 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv4_2_GN')(conv4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
    x = LeakyReLU(alpha=0.1, name='L14')(x)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = Conv3D(64, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
    pool4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_pool4_GN')(pool4)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
    pool4 = LeakyReLU(alpha=0.1, name='L14_p')(pool4)

    # 8*8
    conv5 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool4)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L15')(x)
    conv5 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
    x = LeakyReLU(alpha=0.1, name='L16')(x)
    # drop5 = Dropout(0.5)(conv5)

    # 8*8
    up1 = Conv3DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    up1 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv1_1_GN')(up1)
    up1 = LeakyReLU(alpha=0.1, name='L15_u')(up1)
    merge6 = merge([conv4, up1], mode='concat', concat_axis=-1)
    conv6 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(merge6)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_1_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L17')(x)
    conv6 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv6_2_GN')(conv6)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
    x = LeakyReLU(alpha=0.1, name='L18')(x)

    # 16*16
    up2 = Conv3DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    up2 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv2_1_GN')(up2)
    up2 = LeakyReLU(alpha=0.1, name='L18_u')(up2)
    merge7 = merge([conv3, up2], mode='concat', concat_axis=-1)
    conv7 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(merge7)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_1_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
    x = LeakyReLU(alpha=0.1, name='L19')(x)
    conv7 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv7_2_GN')(conv7)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
    x = LeakyReLU(alpha=0.1, name='L20')(x)

    # 32*32
    up3 = Conv3DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    up3 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv3_1_GN')(up3)
    up3 = LeakyReLU(alpha=0.1, name='L20_u')(up3)
    merge8 = merge([conv2, up3], mode='concat', concat_axis=-1)
    conv8 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(merge8)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_1_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L21')(x)
    conv8 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv8_2_GN')(conv8)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
    x = LeakyReLU(alpha=0.1, name='L22')(x)

    # 64*64
    up4 = Conv3DTranspose(8, kernel_size=3, strides=2, padding='same')(x)
    up4 = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_deconv4_1_GN')(up4)
    up4 = LeakyReLU(alpha=0.1, name='L22_u')(up4)
    merge9 = merge([conv1, conv1_2, up4], mode='concat', concat_axis=-1)
    conv9 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(merge9)
    # x = BatchNormalization(axis=-1, name='entry_flow_conv9_1_GN')(conv9)
    x = GroupNormalization(groups=groups, axis=-1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
    x1 = LeakyReLU(alpha=0.1, name='L23')(x)
    conv9_2 = Conv3D(8, 3,  padding='same', kernel_initializer='he_normal')(merge9)
    x = BatchNormalization(axis=-1, name='entry_flow_conv9_2_GN')(conv9_2)
    x2 = LeakyReLU(alpha=0.1,name='L24')(x)

    # 128*128
    conv10 = Conv3D(1, 1, kernel_initializer='RandomNormal', activation='sigmoid')(x1)

    conv10_2 = Conv3D(1, 1, kernel_initializer='RandomNormal', activation='sigmoid')(x2)

    model = Model(input=[input1, input2], output=[conv10,conv10_2])

    # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])

    return model


#
# def RealThreeDNet_multi_small_laji_code_decode(input_shape=(None, None,None, 1)):
#
#     groups = 2
#
#     input1 = Input(shape=(input_shape[0], input_shape[1],input_shape[2], 1), name='input1')
#     input2 = Input(shape=(input_shape[0], input_shape[1],input_shape[2], 1), name='input2')
#
#     # 128*128*64
#     conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input1)
#     # print("conv1 shape:", conv1.shape)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_GN')(conv1)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_GN')(conv1)
#     x = LeakyReLU(alpha=0.1,name='L1')(x)
#     conv1 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(x)
#     # print("conv1 shape:", conv1.shape)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv1_2_GN')(conv1)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv1_2_GN')(conv1)
#     x = LeakyReLU(alpha=0.1,name='L2')(x)
#     pool1 = Conv3D(8, 3, strides=2,padding='same', kernel_initializer='he_normal')(x)
#     pool1 = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_pool1_GN')(pool1)
#     pool1 = LeakyReLU(alpha=0.1,name='L1_p')(pool1)
#
#
#     conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(input2)
#     # print("conv1 shape:", conv1.shape)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv1_1_2_GN')(conv1_2)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_2_GN')(conv1_2)
#     x = LeakyReLU(alpha=0.1, name='L1_2')(x)
#     pool0 = Conv3D(16, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
#     pool0 = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_pool0_GN')(pool0)
#     pool0 = LeakyReLU(alpha=0.1,name='L0_p')(pool0)
#     deconv1_decode = Conv3DTranspose(16, kernel_size=3, strides=2, padding='same')(pool0)
#     # print("conv1 shape:", conv1.shape)
#     conv1_2 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(deconv1_decode)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv1_2_2_GN')(conv1_2)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv1_1_2_GN')(conv1_2)
#     x = LeakyReLU(alpha=0.1, name='L1_2_2')(x)
#     pool1_2 = Conv3D(8, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
#     pool1_2 = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_pool1_2_GN')(pool1_2)
#     pool1_2 = LeakyReLU(alpha=0.1, name='L1_2_p')(pool1_2)
#
#     concat_conv12 = concatenate([conv1,conv1_2],axis=-1)
#
#     conv_out_1 = Conv3D(1, 1, strides=1, padding='same', kernel_initializer='he_normal',activation='sigmoid')(concat_conv12)
#
#     concat_t1t2 = concatenate([pool1,pool1_2],axis=-1)
#
#
#
#     # 128*128*64
#     conv2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(concat_t1t2)
#     # print("conv2 shape:", conv2.shape)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv2_1_GN')(conv2)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv2_1_GN')(conv2)
#     x = LeakyReLU(alpha=0.1,name='L3')(x)
#     conv2 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
#     # print("conv2 shape:", conv2.shape)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv2_2_GN')(conv2)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
#     x = LeakyReLU(alpha=0.1,name='L4')(x)
#     pool2 = Conv3D(32, 3, strides=2,padding='same', kernel_initializer='he_normal')(x)
#     pool2 = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_pool2_GN')(pool2)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
#     pool2 = LeakyReLU(alpha=0.1,name='L4_p')(pool2)
#     # print("pool2 shape:", pool2.shape)
#
#     # 128*128
#     conv3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool2)
#     # print("conv3 shape:", conv3.shape)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv3_1_GN')(conv3)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv3_1_GN')(conv3)
#     x = LeakyReLU(alpha=0.1,name='L5')(x)
#     conv3 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
#     # print("conv3 shape:", conv3.shape)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv3_2_GN')(conv3)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv3_2_GN')(conv3)
#     x = LeakyReLU(alpha=0.1,name='L6')(x)
#     pool3 = Conv3D(64, 3, strides=2,padding='same', kernel_initializer='he_normal')(x)
#     pool3 = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_pool3_GN')(pool3)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
#     pool3 = LeakyReLU(alpha=0.1,name='L6_p')(pool3)
#
#     # 64*64
#     conv4 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool3)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv4_1_GN')(conv4)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv4_1_GN')(conv4)
#     x = LeakyReLU(alpha=0.1,name='L13')(x)
#     conv4 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv4_2_GN')(conv4)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv4_2_GN')(conv4)
#     x = LeakyReLU(alpha=0.1,name='L14')(x)
#     # drop4 = Dropout(0.5)(conv4)
#     pool4 = Conv3D(64, 3, strides=2,padding='same', kernel_initializer='he_normal')(x)
#     pool4 = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_pool4_GN')(pool4)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv2_2_GN')(conv2)
#     pool4 = LeakyReLU(alpha=0.1,name='L14_p')(pool4)
#
#     # 8*8
#     conv5 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(pool4)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv5_1_GN')(conv5)
#     x = LeakyReLU(alpha=0.1,name='L15')(x)
#     conv5 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv5_2_GN')(conv5)
#     x = LeakyReLU(alpha=0.1,name='L16')(x)
#     # drop5 = Dropout(0.5)(conv5)
#
#     # 8*8
#     up1 = Conv3DTranspose(64,kernel_size=3,strides=2,padding='same')(x)
#     up1 = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_deconv1_1_GN')(up1)
#     up1 = LeakyReLU(alpha=0.1,name='L15_u')(up1)
#     merge6 = merge([conv4, up1], mode='concat', concat_axis=-1)
#     conv6 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(merge6)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv6_1_GN')(conv6)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv6_1_GN')(conv6)
#     x = LeakyReLU(alpha=0.1,name='L17')(x)
#     conv6 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv6_2_GN')(conv6)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv6_2_GN')(conv6)
#     x = LeakyReLU(alpha=0.1,name='L18')(x)
#
#     # 16*16
#     up2 = Conv3DTranspose(64,kernel_size=3,strides=2,padding='same')(x)
#     up2 = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_deconv2_1_GN')(up2)
#     up2 = LeakyReLU(alpha=0.1,name='L18_u')(up2)
#     merge7 = merge([conv3, up2], mode='concat', concat_axis=-1)
#     conv7 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(merge7)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv7_1_GN')(conv7)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv7_1_GN')(conv7)
#     x = LeakyReLU(alpha=0.1,name='L19')(x)
#     conv7 = Conv3D(64, 3, padding='same', kernel_initializer='he_normal')(x)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv7_2_GN')(conv7)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv7_2_GN')(conv7)
#     x = LeakyReLU(alpha=0.1,name='L20')(x)
#
#     # 32*32
#     up3 = Conv3DTranspose(32,kernel_size=3,strides=2,padding='same')(x)
#     up3 = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_deconv3_1_GN')(up3)
#     up3 = LeakyReLU(alpha=0.1,name='L20_u')(up3)
#     merge8 = merge([conv2, up3], mode='concat', concat_axis=-1)
#     conv8 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(merge8)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv8_1_GN')(conv8)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv8_1_GN')(conv8)
#     x = LeakyReLU(alpha=0.1,name='L21')(x)
#     conv8 = Conv3D(32, 3, padding='same', kernel_initializer='he_normal')(x)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv8_2_GN')(conv8)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv8_2_GN')(conv8)
#     x = LeakyReLU(alpha=0.1,name='L22')(x)
#
#     # 64*64
#     up4 = Conv3DTranspose(8,kernel_size=3,strides=2,padding='same')(x)
#     up4 = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_deconv4_1_GN')(up4)
#     up4 = LeakyReLU(alpha=0.1,name='L22_u')(up4)
#     merge9 = merge([conv1,conv1_2, up4], mode='concat', concat_axis=-1)
#     conv9 = Conv3D(8, 3, padding='same', kernel_initializer='he_normal')(merge9)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv9_1_GN')(conv9)
#     x = GroupNormalization(groups=4, axis=-1, epsilon=0.1, name='entry_flow_conv9_1_GN')(conv9)
#     x = LeakyReLU(alpha=0.1,name='L23')(x)
#     # conv9 = Conv3D(16, 3,  padding='same', kernel_initializer='he_normal')(x)
#     # x = BatchNormalization(axis=-1, name='entry_flow_conv9_2_GN')(conv9)
#     # x = LeakyReLU(alpha=0.1,name='L24')(x)
#
#     # 128*128
#     conv10 = Conv3D(1, 1,kernel_initializer='RandomNormal',activation='sigmoid')(x)
#
#     model = Model(input=[input1,input2], output=[conv10,conv_out_1])
#
#     # model.compile(optimizer=SGD(lr=1.0e-8), loss=EuclideanLoss, metrics=['acc',precision,recall,fmeasure,tp,fp,tn,fn,yt_sum])
#
#     return model
#
