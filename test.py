from __future__ import absolute_import

from got10k.experiments import *

from siamfc import TrackerSiamFC
from config import config
import os

if __name__ == '__main__':
    name = 'SoftMaskSiam_135_263'

    # setup tracker
    net_path = 'model2/model_e31.pth'

    epoch_num = 50
    # training loop
    for epoch in range(24, epoch_num):
        result_dir=os.path.join('results3/SoftMaskSiam_135_263/SiamFC_Epoc_0_{}'.format(epoch + 1))
        report_dir=os.path.join('reports3/SoftMaskSiam_135_263/SiamFC_Epoc_0_{}'.format(epoch + 1))
        experiments = ExperimentOTB(config.root_dir_for_OTB, version=2015, 
                                    result_dir=result_dir , report_dir = report_dir)
#
#        # run tracking experiments and report performance
        #res_name='SiamFC_Epoc_{}'.format(epoch+1)
        net_path= 'model3/model_e{}.pth'.format(epoch+1)
        tracker = TrackerSiamFC(net_path=net_path)
        experiments.run(tracker)
        experiments.report([tracker.name])