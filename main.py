'''
DCGAN - CIFAR-10 Color Photographs.

Ref: https://machinelearningmastery.com/generative_adversarial_networks/
'''

from data import loadRealSamples
from gan import createGan, train
from generator import createGenerator
from discriminator import createDiscriminator

if __name__ == '__main__':
    latentDim = 100
    dataset = loadRealSamples()
    discriminator = createDiscriminator()
    generator = createGenerator(latentDim)
    gan = createGan(discriminator, generator)
    train(discriminator, generator, gan, dataset, latentDim)