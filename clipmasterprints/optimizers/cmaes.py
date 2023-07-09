import cma
import numpy as np
import pickle

import logging

class CMAESOptimizer:

    @staticmethod
    def get_default_dict():
        return {'save_after_x_iter': 1000, 'init_vector':'norm', 'sigma_0': 1., 'pop_size':'default', 'max_iter': 'forever'}

    def construct_hyperparam_dict(self, hyperparams):
        return_dict = CMAESOptimizer.get_default_dict()
        if hyperparams is not None:
            for (key, value) in hyperparams.items():
                return_dict[key] = value
        return return_dict

    def __init__(self, loss, latent_dims, hyperparams=None,save_callback=None, device=None):
        #NOTE: device is a dummy parameter for interface compatiblity here, and is currently unused
        self.loss = loss
        self.hyperparams = self.construct_hyperparam_dict(hyperparams)

        if save_callback is not None:
            self.save_callback = save_callback
        else:
            def save_default(es_object,iter):
                with open(f'es_iter_{iter}.pkl','wb') as pkl_file:
                    pickle.dump(es_object,pkl_file)
            self.save_callback = save_default

        self.n_features = np.prod(latent_dims)
        self.max_iter = self.hyperparams['max_iter'] if self.hyperparams['max_iter'] != 'forever' else None
        self.iter = 0
        if self.hyperparams['init_vector'] == 'norm':
            init_vector = np.random.randn(self.n_features,)
        else:
            init_vector = self.hyperparams.init_vector

        self.es = cma.CMAEvolutionStrategy(init_vector, self.hyperparams['sigma_0'])

    def log(self, message=None):
        if message == None:
            self.es.disp()
        else:
            logging.debug(message)

    def finished(self):
        if self.max_iter is not None:
            return self.iter > self.max_iter
        return self.es.stop()

    def save(self):
        self.save_callback(self.es,self.iter)

    def step(self):
        X = self.es.ask()  # sample len(X) candidate solutions
        losses = self.loss.loss(X)
        self.es.tell(X, losses)
        # TODO: check how to update this without error
        # cfun.update(es)
        self.es.logger.add()  # for later plotting

    def optimize(self):
        while not self.finished():

            self.step()
            self.log()

            if (self.iter % self.hyperparams['save_after_x_iter']) == 0:
                self.save()

            self.iter += 1
        self.save()
        self.es.result_pretty()
