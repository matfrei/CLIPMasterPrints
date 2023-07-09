import os
import time
from pathlib import Path

class Experiment():
    def __init__(self,basepath, model_type, experiment_name, ts_dirname=None):
        self.time_fstring = "%d-%m-%Y_%H-%M-%S"
        if ts_dirname is not None:
            self.ts_dirname=ts_dirname
            self.timestamp = None
        else: 
            self.timestamp = time.localtime()
            self.ts_dirname = time.strftime(self.time_fstring,self.timestamp)
        
        self.basepath = basepath
        self.model_type = model_type
        self.experiment_name = experiment_name
            
        self.modelpath = os.path.join(basepath,'models',model_type,experiment_name,self.ts_dirname)
        self.outputpath = os.path.join(basepath,'output',model_type,experiment_name,self.ts_dirname)
        Path(self.basepath).mkdir(parents=True, exist_ok=True)

    def weight_path(self):
        ret_path = os.path.join(self.modelpath,'weights')
        Path(self.modelpath).mkdir(parents=True, exist_ok=True)
        Path(ret_path).mkdir(parents=True, exist_ok=True)
        return ret_path
    
    def tb_path(self):
        ret_path = os.path.join(self.modelpath,'tensorboard')
        Path(self.modelpath).mkdir(parents=True, exist_ok=True)
        Path(ret_path).mkdir(parents=True, exist_ok=True)
        return ret_path
        
    def log_path(self):
        ret_path =  os.path.join(self.modelpath,'log')
        Path(self.modelpath).mkdir(parents=True, exist_ok=True)
        Path(ret_path).mkdir(parents=True, exist_ok=True)
        return ret_path
    
    def output_path(self,timestamp=True):
        if timestamp:
            parent_dirname = self.ts_dirname
        else:
            parent_dirname = 'final'
        self.outputpath = os.path.join(self.basepath,'output',self.model_type,self.experiment_name,parent_dirname)  
        Path(self.outputpath).mkdir(parents=True, exist_ok=True) 
        return self.outputpath
