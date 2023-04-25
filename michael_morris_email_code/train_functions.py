import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

tfd = tfp.distributions

@tf.function
def split_uneven(x, n, axis=0):
    shape = tf.shape(x)[axis] 
    size_of_splits = tf.math.floor(shape / n)
    new_size = tf.cast(size_of_splits * n, tf.int32)

    ls = tf.split(x[:new_size], n, axis)
    ls.append(x[new_size:])
    return ls

def fit(_model, dataset, optimizer, loss_fn, epochs = 1, split_steps = 3, prediction_steps =3, speedy_training=False, verbose=True):
    @tf.function
    def train_step(_model, x, y, prediction_steps=prediction_steps, split_steps=split_steps, teacher_forcing=True):
        with tf.GradientTape() as tape:
            nll = []
            for xn, yn in zip(split_uneven(x, split_steps), split_uneven(y, split_steps)):
                if not speedy_training:
                    pred = [_model(xn, training = True, teacher_forcing = teacher_forcing) for _ in range(prediction_steps)]

                    means = tf.convert_to_tensor([p.mean() for p in pred])
                    vars = tf.convert_to_tensor([p.variance() for p in pred])
                else:
                    xn2 = tf.tile(xn, (prediction_steps,1,1))
                    yn_pred = _model(xn2, teacher_forcing = teacher_forcing)

                    means = tf.reshape(yn_pred.mean(), (prediction_steps, ) + yn.shape)
                    vars = tf.reshape(yn_pred.variance(), (prediction_steps, ) + yn.shape)

                mean = tf.reduce_mean(means, 0)
                std = tf.math.sqrt(tf.reduce_mean(vars, 0) + tf.math.reduce_variance(means, 0))

                y_hat = tfp.distributions.Normal(mean, std)
                nll.append(loss_fn(yn, y_hat))
            
            nll = tf.reduce_mean(tf.concat(nll, 0))
            kl = tf.reduce_mean(tf.concat([_model.layers[0].KL_rec_kernel,
                                           _model.layers[0].KL_kernel,
                                           _model.layers[0].KL_bias, 
                                           _model.layers[2].kl_penalty
                                          ], 0))

            loss_value = kl + nll
        grads = tape.gradient(loss_value, _model.trainable_weights)
        optimizer.apply_gradients(zip(grads, _model.trainable_weights))
        return nll, kl


    history = []
    for epoch in range(epochs):
        nll_total = 0
        kl_total = 0

        with tqdm(dataset, unit="batch") as tepoch:
            step = 1
            for x_batch_train, y_batch_train in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")

                nll_value, kl_value = train_step(_model, x_batch_train, y_batch_train)
                nll_total += tf.reduce_mean(nll_value).numpy()
                kl_total += tf.reduce_mean(kl_value).numpy()

                tepoch.set_postfix(nll=nll_total/step, kl=kl_total/step)
                step+=1
        # for step, (x_batch_train, y_batch_train) in enumerate(dataset):
        #     nll_value, kl_value = train_step(_model, x_batch_train, y_batch_train)
        #     nll_total += tf.reduce_mean(nll_value).numpy()
        #     kl_total += tf.reduce_mean(kl_value).numpy()

        #     if verbose:
        #         print('Epoch:',epoch+1, '\tNLL:','{0:.2f}'.format(nll_total/(step+1)), 'KL:','{0:.2f}'.format(kl_total/(step+1)), end='\r')
        # if verbose:
        #     print('Epoch:',epoch+1, '\tNLL:','{0:.2f}'.format(nll_total/(step+1)), 'KL:','{0:.2f}'.format(kl_total/(step+1)))
        history.append({'nll': nll_total / (step + 1), 'kl': kl_total / (step + 1)})
    return _model, history