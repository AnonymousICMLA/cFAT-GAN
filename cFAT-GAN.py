#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import re
from keras.layers.merge import _Merge, concatenate, dot
from keras.layers.normalization import BatchNormalization
from matplotlib import colors as mcol
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from functools import partial
from keras.models import Model, Sequential, model_from_json
from keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    Dropout,
    ActivityRegularization,
    Lambda,
    Concatenate,
    Permute,
    Convolution1D,
    MaxPooling1D,
    AveragePooling1D,
    GlobalAveragePooling1D
)

# set up environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
imagedir = 'dump/'


# 4 dimensional scalar product
def vec4dot(v1, v2):
    term0 = v1[0]*v2[0]
    term1 = v1[1]*v2[1]
    term2 = v1[2]*v2[2]
    term3 = v1[3]*v2[3]
    return term0 - term1 - term2 - term3


# 4-vector for electron beam
def get_electron_vec(BEAM_ENERGY):
    beam_electron_vec = [BEAM_ENERGY, 0.0, 0.0, BEAM_ENERGY]
    return beam_electron_vec


# 4-vector for proton beam
def get_proton_vec(BEAM_ENERGY):
    beam_proton_vec = [BEAM_ENERGY, 0.0,
                       0.0, -np.sqrt(BEAM_ENERGY ** 2.0 - 0.938 * 0.938)]
    return beam_proton_vec


# read and process pythia event data files at the training energies
def get_samples(pevent_data_files, TRAIN_ENERGIES):
    peventnum = 0
    electron = np.empty([1000000, 11])
    Q2min = 100000.0
    for i in range(len(TRAIN_ENERGIES)):
        beamEnergy = TRAIN_ENERGIES[i]
        beamElectron4vec = get_electron_vec(beamEnergy)
        beamProton4vec = get_proton_vec(beamEnergy)
        with open(pevent_data_files[i], "r") as fp:
            line = fp.readline()
            while line:
                particle = re.split(' +', line)
                px = float(particle[1])
                py = float(particle[2])
                pz = float(particle[3])
                pxy = px*py
                pxz = px*pz
                pyz = py*pz
                pt = np.sqrt(px*px+py*py)
                e = np.sqrt(px*px+py*py+pz*pz)
                pzt = pz/(pt+0.01)

                q = [i-j for i, j in zip(beamElectron4vec, [e, px, py, pz])]
                Q2 = -vec4dot(q, q)
                if Q2 > 1.0:
                    lnz = np.log(beamEnergy - pz)
                    electron[peventnum] = np.array([beamEnergy, px, py,
                                                    lnz, pxy, pxz, pyz,
                                                    pt, e, pz, pzt])
                    peventnum += 1
                    if Q2 < Q2min:
                        Q2min = Q2
                line = fp.readline()

    electron = electron[:peventnum]
    return electron, Q2min


def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate.
    In a standard GAN, the discriminator has a sigmoid output,
    representing the probability that samples are real or generated.
    In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1], the
    discriminator wants to make the distance between its output for
    real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples
    -1 and real samples 1, instead of the 0 and 1 used in normal GANs,
    so that multiplying the outputs by the labels will give you the loss
    immediately.Note that the nature of this loss means that it can be
    (and frequently will be)less than 0."""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding
    a term to the loss function that penalizes the network if the gradient
    norm moves away from 1. However, it is impossible to evaluate this
    function at all points in the input space. The compromise used in the
    paper is to choose random points on the lines between real and generated
    samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through
    the generator and evaluate the loss. Then we get the gradients of the
    discriminator w.r.t. the input averaged samples. The l2 norm and penalty
    can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as
    input, but Keras only supports passing y_true and y_pred to loss
    functions. To get around this, we make a partial() of the function with
    the averaged_samples argument, and use that for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])


# calculate augmented features: pz,px*py,px*pz,py*pz,e,pt,pz,
# and pz/pt from generated features
def feature_mul(x):
    pxyzmean = K.constant(xyzmean)
    pxyzstd = K.constant(xyzstd)
    xyz = x*pxyzstd+pxyzmean
    beamEnergy = xyz[:, 0:1]
    px = xyz[:, 1:2]
    py = xyz[:, 2:3]
    lnz = xyz[:, 3:4]
    pz = beamEnergy - K.exp(lnz)

    pxsq = px*px
    pysq = py*py
    pzsq = pz*pz

    pxy = px*py
    pxz = px*pz
    pyz = py*pz

    pxyz = K.concatenate([pxy, pxz, pyz])
    pxyz = (pxyz - K.constant(xyz2mean))/K.constant(xyz2std)
    pt = K.sqrt(pxsq+pysq)
    pzt = pz/(pt+0.01)
    e = (K.sqrt(pxsq+pysq+pzsq) - K.constant(emean))/K.constant(estd)
    pt = (pt - K.constant(ptmean))/K.constant(ptstd)
    pz = (pz - K.constant(pzmean))/K.constant(pzstd)
    pzt = (pzt - K.constant(pztmean))/K.constant(pztstd)

    return K.concatenate([pxyz, pt, e, pz, pzt])


# define generator
# output is beamEnergy, px,py,T(pz),pxyz,pt,e,pz,pzt
# which is a 11 dimensional vector
def make_generator():
    beam = Input(shape=(1,))
    noise = Input(shape=(100,))
    visible = concatenate([beam, noise])
    hidden1 = Dense(512)(visible)
    LR = LeakyReLU(alpha=0.2)(hidden1)
    hidden2 = Dense(512)(LR)
    LR = LeakyReLU(alpha=0.2)(hidden2)
    hidden3 = Dense(512)(LR)
    LR = LeakyReLU(alpha=0.2)(hidden3)
    hidden4 = Dense(512)(LR)
    LR = LeakyReLU(alpha=0.2)(hidden4)
    hidden5 = Dense(512)(LR)
    LR = LeakyReLU(alpha=0.2)(hidden5)
    hidden6 = Dense(512)(LR)
    LR = LeakyReLU(alpha=0.2)(hidden6)
    hidden7 = Dense(512)(LR)
    LR = LeakyReLU(alpha=0.2)(hidden7)
    hidden8 = Dense(512)(LR)
    LR = LeakyReLU(alpha=0.2)(hidden8)
    output = Dense(3)(LR)
    output2 = concatenate([beam, output])
    feature = Lambda(feature_mul)(output2)

    outputmerge = concatenate([output2, feature])

    generator = Model(inputs=[beam, noise], outputs=[outputmerge])
    return(generator)


# define discrimnator using leakyReLU and dropout for all layers
def make_discriminator():
    visible = Input(shape=(11,))
    hidden1 = Dense(512)(visible)
    LR = LeakyReLU(alpha=0.2)(hidden1)
    DR = Dropout(rate=0.1)(LR)
    hidden2 = Dense(512)(DR)
    LR = LeakyReLU(alpha=0.2)(hidden2)
    DR = Dropout(rate=0.1)(LR)
    hidden3 = Dense(512)(DR)
    LR = LeakyReLU(alpha=0.2)(hidden3)
    DR = Dropout(rate=0.1)(LR)
    hidden4 = Dense(512)(DR)
    LR = LeakyReLU(alpha=0.2)(hidden4)
    DR = Dropout(rate=0.1)(LR)
    hidden5 = Dense(512)(DR)
    LR = LeakyReLU(alpha=0.2)(hidden5)
    DR = Dropout(rate=0.1)(LR)
    hidden5 = Dense(512)(DR)
    LR = LeakyReLU(alpha=0.2)(hidden5)
    DR = Dropout(rate=0.1)(LR)
    hidden5 = Dense(512)(DR)
    LR = LeakyReLU(alpha=0.2)(hidden5)
    DR = Dropout(rate=0.1)(LR)
    hidden5 = Dense(512)(DR)
    LR = LeakyReLU(alpha=0.2)(hidden5)
    DR = Dropout(rate=0.1)(LR)
    output = Dense(1)(DR)

    discriminator = Model(inputs=[visible], outputs=output)
    return discriminator


def train(electron):
    generator = make_generator()
    discriminator = make_discriminator()

    for layer in discriminator.layers:
        layer.trainable = False
    discriminator.trainable = False

    generator_beam = Input(shape=(1,))
    generator_noise = Input(shape=(100,))
    generator_input = generator([generator_beam, generator_noise])
    discriminator_layers_for_generator = discriminator(generator_input)
    generator_model = Model(inputs=[generator_beam, generator_noise],
                            outputs=[discriminator_layers_for_generator])
    # We use the Adam paramaters from Gulrajani et al.
    generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss])
    generator_model.summary()

    for layer in discriminator.layers:
        layer.trainable = True
    for layer in generator.layers:
        layer.trainable = False
    discriminator.trainable = True
    generator.trainable = False

    real_samples = Input(shape=(11,))
    generator_beam_for_discriminator = Input(shape=(1,))
    generator_noise_for_discriminator = Input(shape=(100,))
    generated_samples_for_discriminator = generator(
        [generator_beam_for_discriminator, generator_noise_for_discriminator])
    discriminator_output_from_generator = discriminator(
        generated_samples_for_discriminator)
    discriminator_output_from_real_samples = discriminator(real_samples)

    # We also need to generate weighted-averages of real and generated samples,
    # to use for the gradient norm penalty.
    averaged_samples = RandomWeightedAverage()(
        [real_samples, generated_samples_for_discriminator])

    # We then run these samples through the discriminator as well.
    # Note that we never really use the discriminator output for
    # these samples - we're only running them to get the gradient
    # norm for the gradient penalty loss.
    averaged_samples_out = discriminator(averaged_samples)

    # The gradient penalty loss function requires the input
    # averaged samples to get gradients. However, Keras loss
    # functions can only have two arguments, y_true and y_pred.
    # We get around this by making a partial() of the function with
    # the averaged samples here.
    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
    # Functions need names or Keras will throw an error
    partial_gp_loss.__name__ = 'gradient_penalty'

    # However, if we don't concatenate the real and generated samples,
    # we get three outputs: One of the generated samples, one of the real
    # samples, and one of the averaged samples, all of size BATCH_SIZE.
    discriminator_model = Model(
        inputs=[real_samples,
                generator_beam_for_discriminator,
                generator_noise_for_discriminator],
        outputs=[discriminator_output_from_real_samples,
                 discriminator_output_from_generator,
                 averaged_samples_out])
    # We use the Adam paramaters from Gulrajani et al.
    # We use the Wasserstein loss for both the real and
    # generated samples, and the gradient penalty loss for the averaged samples
    discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                                loss=[wasserstein_loss,
                                      wasserstein_loss,
                                      partial_gp_loss])
    discriminator_model.summary()

    # We make three label vectors for training.
    # positive_y is the label vector for real samples, with value 1.
    # negative_y is the label vector for generated samples, with
    # value -1. The dummy_y vector is passed to the gradient_penalty
    # loss function and is not used.
    positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
    negative_y = -positive_y
    dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

    loss = []
    for epoch in range(100000):
        np.random.shuffle(electron)
        # print("Number of batches: ", int(dataset.shape[0] // BATCH_SIZE))
        discriminator_loss = []
        generator_loss = []
        minibatches_size = BATCH_SIZE * TRAINING_RATIO
        for i in range(int(electron.shape[0] // (minibatches_size))):
            discriminator_minibatches = electron[i * minibatches_size:
                                                 (i + 1) * minibatches_size]

            noise = np.random.normal(0, 1, [BATCH_SIZE*TRAINING_RATIO, 100])
            beam = np.random.randint(low=1, high=6,
                                     size=[BATCH_SIZE*TRAINING_RATIO, 1])*10
            beam = (beam - beamEnergymean)/beamEnergystd
            for j in range(TRAINING_RATIO):
                image_batch = discriminator_minibatches[j * BATCH_SIZE:
                                                        (j + 1) * BATCH_SIZE]
                noise_batch = noise[j * BATCH_SIZE:
                                    (j + 1) * BATCH_SIZE]
                beam_batch = beam[j * BATCH_SIZE:
                                  (j + 1) * BATCH_SIZE]
                discriminator_loss.append(discriminator_model.train_on_batch(
                    [image_batch, beam_batch, noise_batch],
                    [positive_y, negative_y, dummy_y]))

            noise = np.random.normal(0, 1, [BATCH_SIZE, 100])
            beam = np.random.randint(low=1, high=6, size=[BATCH_SIZE, 1])*10
            beam = (beam - beamEnergymean)/beamEnergystd
            generator_loss.append(generator_model.train_on_batch(
                [beam, noise], [positive_y]))
        print(epoch, generator_loss)
        loss.append([epoch, generator_loss])

        if epoch % 500 == 0:
            f = open(imagedir+"loss.txt", "w+")
            f.write(str(loss))
            f.close()

        generator.save_weights(
            imagedir+'generator'+str(epoch//100).zfill(5)+'.h5')


def gen_samples(weight_path, SAMPLE_SIZE,
                energies=[10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90]):
    generator = make_generator()
    generator.load_weights(weight_path)
    # set up dictionaries to hold generated samples
    eDict = {}
    eDict['Generated'] = {}
    # generate samples at the five trained energies
    for i in range(len(energies)):
        b = energies[i]
        eDict['Generated'][str(b)] = [[], [], [], [], [], [], []]
        noise = np.random.normal(0, 1, [SAMPLE_SIZE, 100])
        beam = np.ones([SAMPLE_SIZE, 1])*float(b)
        beam = (beam - beamEnergymean)/beamEnergystd
        results = generator.predict([beam, noise])
        # load results into dictionary
        beamElectron4vec = get_electron_vec(b)
        beamProton4vec = get_proton_vec(b)
        for ii in range(results.shape[0]):
            pxyz = results[ii][0:4]*xyzstd+xyzmean
            beamEnergy = pxyz[0]
            px = pxyz[1]
            py = pxyz[2]
            lnz = pxyz[3]
            pz = (beamEnergy - np.exp(lnz))
            pt = np.sqrt(px*px + py*py)
            e = np.sqrt(px*px + py*py + pz*pz)
            q = [i-j for i, j in zip(beamElectron4vec, [e, px, py, pz])]
            Q2 = -vec4dot(q, q)
            if Q2 > Q2min and e < beamEnergy and pz > 1.0:
                xbj = Q2 / (2.0 * vec4dot(q, beamProton4vec))
                eDict['Generated'][str(b)][0].append(e)
                eDict['Generated'][str(b)][1].append(px)
                eDict['Generated'][str(b)][2].append(py)
                eDict['Generated'][str(b)][3].append(pz)
                eDict['Generated'][str(b)][4].append(pt)
                eDict['Generated'][str(b)][5].append(Q2)
                eDict['Generated'][str(b)][6].append(xbj)
    gen_dict = json.dumps(eDict)
    f = open('data/for_plot/eDict.json', 'w')
    f.write(gen_dict)
    f.close

# hardcode training parameters
HALF_BATCH = 4000
BATCH_SIZE = HALF_BATCH * 2

TRAINING_RATIO = 5
GRADIENT_PENALTY_WEIGHT = 10

TRAIN_ENERGIES = [10.0, 20.0, 30.0, 40.0, 50.0]
pevent_data_files = ["data/train_data/tape_10", "data/train_data/tape_20",
                     "data/train_data/tape_30", "data/train_data/tape_40",
                     "data/train_data/tape_50"]

electron, Q2min = get_samples(pevent_data_files, TRAIN_ENERGIES)

# normalization
electronmean = np.mean(electron, axis=0)
electronstd = np.std(electron, axis=0)
beamEnergymean = electronmean[0]
beamEnergystd = electronstd[0]
xyzmean = electronmean[0:4]
xyzstd = electronstd[0:4]
xyz2mean = electronmean[4:7]
xyz2std = electronstd[4:7]
ptmean = electronmean[7]
ptstd = electronstd[7]
emean = electronmean[8]
estd = electronstd[8]
pzmean = electronmean[9]
pzstd = electronstd[9]
pztmean = electronmean[10]
pztstd = electronstd[10]

# normalized training set
electron = (electron - electronmean)/electronstd


def main():
    train(electron)

main()
