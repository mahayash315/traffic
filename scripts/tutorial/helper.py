import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict


def get_adam_optimizer(learning_rate=0.001, decay1=0.1, decay2=0.001, weight_decay=0.0):
    print 'AdaM', learning_rate, decay1, decay2, weight_decay
    def shared32(x, name=None, borrow=False):
        return theano.shared(np.asarray(x, dtype='float32'), name=name, borrow=borrow)

    def get_optimizer(w, g):
        updates = OrderedDict()
        
        it = shared32(0.)
        updates[it] = it + 1.
        
        fix1 = 1.-(1.-decay1)**(it+1.) # To make estimates unbiased
        fix2 = 1.-(1.-decay2)**(it+1.) # To make estimates unbiased
        lr_t = learning_rate * T.sqrt(fix2) / fix1
        
        for i in w:
    
            gi = g[i]
            if weight_decay > 0:
                gi -= weight_decay * w[i] #T.tanh(w[i])

            # mean_squared_grad := E[g^2]_{t-1}
            mom1 = shared32(w[i].get_value() * 0.)
            mom2 = shared32(w[i].get_value() * 0.)
            
            # Update moments
            mom1_new = mom1 + decay1 * (gi - mom1)
            mom2_new = mom2 + decay2 * (T.sqr(gi) - mom2)
            
            # Compute the effective gradient and effective learning rate
            effgrad = mom1_new / (T.sqrt(mom2_new) + 1e-10)
            
            effstep_new = lr_t * effgrad
            
            # Do update
            w_new = w[i] + effstep_new
                
            # Apply update
            updates[w[i]] = w_new
            updates[mom1] = mom1_new
            updates[mom2] = mom2_new
            
        return updates
    
    return get_optimizer