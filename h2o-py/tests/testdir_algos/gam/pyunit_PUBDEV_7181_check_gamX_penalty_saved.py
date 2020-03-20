from __future__ import division
from __future__ import print_function
from past.utils import old_div
import sys
sys.path.insert(1, "../../../")
import h2o
from tests import pyunit_utils
import pandas as pd
import zipfile
import statsmodels.api as sm
from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator


def test_gam_penalty_gamC():
    h2o_data = h2o.import_file(
        path=pyunit_utils.locate("smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv"))
    h2o_data["C1"] = h2o_data["C1"].asfactor()
    h2o_data["C2"] = h2o_data["C2"].asfactor()
    myX = ["C1", "C2"]
    myY = "C11"
    h2o_data["C11"] = h2o_data["C11"].asfactor()

    # h2o_model = H2OGeneralizedAdditiveEstimator(family="binomial", link="logit", gam_X=["C6", "C7", "C8"],
    #                                             saveGamCols=True, savePenaltyMat=True,
    #                                             scale=[1, 1, 1], k=[5, 5, 5],
    #                                             knots=[[-1.99905699, -0.98143075, 0.02599159, 1.00770987, 1.99942290],
    #                                                    [-1.999821861, -1.005257990, -0.006716042, 1.002197392,
    #                                                     1.999073589],
    #                                                    [-1.999675688, -0.979893796, 0.007573327, 1.011437347,
    #                                                     1.999611676]])

    h2o_model = H2OGeneralizedAdditiveEstimator(family="multinomial", gam_X=["C6", "C7", "C8"],
                                                 saveGamCols=True, savePenaltyMat=True,
                                                 scale = [1,1,1])
    h2o_model.train(x=myX, y=myY, training_frame=h2o_data)
    print("Done")


if __name__ == "__main__":
    h2o.init(ip="192.168.86.39", port=54321, strict_version_check=False)
    pyunit_utils.standalone_test(test_gam_penalty_gamC)
else:
    h2o.init(ip="192.168.86.39", port=54321, strict_version_check=False)
    test_gam_penalty_gamC()
