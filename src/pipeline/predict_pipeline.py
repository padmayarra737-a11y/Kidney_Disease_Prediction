import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 age: float,
                 bp: float,
                 sg: float,
                 al: float,
                 su: float,
                 rbc: str,
                 pc: str,
                 pcc: str,
                 ba: str,
                 bgr: float,
                 bu: float,
                 sc: float,
                 sod: float,
                 pot: float,
                 hemo: float,
                 pcv: float,
                 wc: float,
                 rc: float,
                 htn: str,
                 dm: str,
                 cad: str,
                 appet: str,
                 pe: str,
                 ane: str):
        self.age = age
        self.bp = bp
        self.sg = sg
        self.al = al
        self.su = su
        self.rbc = rbc
        self.pc = pc
        self.pcc = pcc
        self.ba = ba
        self.bgr = bgr
        self.bu = bu
        self.sc = sc
        self.sod = sod
        self.pot = pot
        self.hemo = hemo
        self.pcv = pcv
        self.wc = wc
        self.rc = rc
        self.htn = htn
        self.dm = dm
        self.cad = cad
        self.appet = appet
        self.pe = pe
        self.ane = ane

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "bp": [self.bp],
                "sg": [self.sg],
                "al": [self.al],
                "su": [self.su],
                "rbc": [self.rbc],
                "pc": [self.pc],
                "pcc": [self.pcc],
                "ba": [self.ba],
                "bgr": [self.bgr],
                "bu": [self.bu],
                "sc": [self.sc],
                "sod": [self.sod],
                "pot": [self.pot],
                "hemo": [self.hemo],
                "pcv": [self.pcv],
                "wc": [self.wc],
                "rc": [self.rc],
                "htn": [self.htn],
                "dm": [self.dm],
                "cad": [self.cad],
                "appet": [self.appet],
                "pe": [self.pe],
                "ane": [self.ane]
            }  
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            logging.info("Exception occurred in prediction pipeline")
            raise CustomException(e, sys)