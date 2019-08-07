import * as tf from '@tensorflow/tfjs'
import MultivariateNormal from 'multivariate-normal'

class MDNSigmasActivation extends tf.serialization.Serializable {
  /*
  A custom activation, used in the python Keras implementation of
  this MDN model, and required for loading as a Layers Model in tfjs
  */
  static get className () {
    return 'eluPlusOnePlusEpsilon' // its name in the python implementation
  }

  constructor (config) {
    super(config)
    this.epsilonPlusOne = tf.scalar(1.0000001, 'float32')
  }

  getConfig () {
    return {}
  }

  apply (x) {
    // ELU activation, with a small addition to help prevent NaN in loss
    return tf.elu(x).add(this.epsilonPlusOne)
  }

  call (x) {
    return this.apply(x)
  }
}
tf.serialization.registerClass(MDNSigmasActivation)

const splitMixtureParams = (params, outputDim, numMixes) => {
  /*
  Splits up an array of mixture parameters into mus, sigmas, and pis
  depending on the number of mixtures and output dimension.

  Arguments:
  params -- the parameters of the mixture model
  outputDim -- the dimension of the normal models in the mixture model
  numMixes -- the number of mixtures represented
  */
  const totalGaussians = numMixes * outputDim
  const mus = params.slice(0, totalGaussians)
  const sigs = params.slice(totalGaussians, totalGaussians * 2)
  const piLogits = params.slice(params.length - numMixes)
  return [mus, sigs, piLogits]
}

const softmax = (w, t = 1.0) => {
  /*
  Softmax function for a list or numpy array of logits. Also adjusts temperature.

  Arguments:
  w -- a list or numpy array of logits

  Keyword arguments:
  t -- the temperature for to adjust the distribution (default 1.0)
  */

  // adjust temperature
  let e = w.map(el => el / t)
  // subtract max to protect from exploding exp values
  const arrayMax = Math.max(...e)
  e = e.map(el => el - arrayMax)
  e = e.map(el => Math.exp(el))
  const summed = e.reduce((total, val) => total + val, 0)
  const dist = e.map(el => el / summed)
  return dist
}

const sampleFromCategorical = (dist) => {
  /*
  Samples from a categorical model PDF.

  Arguments:
  dist -- the parameters of the categorical model

  Returns:
  One sample from the categorical model
  */
  // const r = Math.random()
  const r = 0.5
  let accumulate = 0
  for (let i = 0; i < dist.length; i++) {
    accumulate += dist[i]
    if (accumulate >= r) {
      return i
    }
  }
  throw new Error('Error sampling categorical model')
}

const createIdentityMatrix = (n, valArray) => {
  /*
  Creates an identity matrix with rows/columns of size "n"
  Optionally, an array of values (of length "n") can be passed
  to set the value at each non-zero point in the identity matrix

  Arguments:
  n -- the dimension of the 2d identity matrix (n * n)
  valArray -- (optional) array of length "n",
              holding the values of the identity matrix

  Returns:
  2d array, representing an identity matrix
  */

  const retVal = []
  for (let x = 0; x < n; x++) {
    retVal.push([])
    for (let y = 0; y < n; y++) {
      if (y === x) {
        if (valArray) {
          retVal[retVal.length - 1].push(valArray[x])
        } else {
          retVal[retVal.length - 1].push(1)
        }
      } else {
        retVal[retVal.length - 1].push(0)
      }
    }
  }
  return retVal
}

export const sampleMDNOutput = (params, outputDim, numMixes, temp = 1.0, sigmaTemp = 1.0) => {
  /*
  Sample from an MDN output with temperature adjustment.
  This calculation is done outside of the Keras model

  Arguments:
  params -- the parameters of the mixture model
  outputDim -- the dimension of the normal models in the mixture model
  numMixes -- the number of mixtures represented

  Keyword arguments:
  temp -- the temperature for sampling between mixture components (default 1.0)
  sigmaTemp -- the temperature for sampling from the normal distribution (default 1.0)

  Returns:
  One sample from the the mixture model
  */
  const [mus, sigs, piLogits] = splitMixtureParams(params, outputDim, numMixes)
  const pis = softmax(piLogits, temp)
  const m = sampleFromCategorical(pis)
  const musVector = mus.slice(m * outputDim, (m + 1) * outputDim)
  const sigVector = sigs.slice(
    m * outputDim,
    (m + 1) * outputDim
  ).map(el => el * sigmaTemp)
  const covMatrix = createIdentityMatrix(outputDim, sigVector)
  const sample = MultivariateNormal(musVector, covMatrix).sample()
  return sample
}
