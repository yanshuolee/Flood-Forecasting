import os

def get(df):
    FP = {}
    FP["trainXFp"] = os.path.join("exp", "train", df, "nor_train_merge_WL_w1.npy")
    FP["trainYFp"] = os.path.join("exp", "train", df, "train_gt_merge_WL_w1.npy")
    FP["testXFp"] = os.path.join("exp", "test", df, "nor_test_merge_WL_w1.npy")
    FP["testYFp"] = os.path.join("exp", "test", df, "test_gt_merge_WL_w1.npy")
    FP["testYWaterInfo"] = os.path.join("exp", "test", df, "test_gt_merge_WL_info.npy")

    # FP["testXFp"] = os.path.join("exp", "test", "bal","bal_nor_test_merge_WL_w1.npy")
    # FP["testYFp"] = os.path.join("exp", "test","bal","bal_test_gt_merge_WL_w1.npy")

    FP["scale_model"] = os.path.join("exp", "scaler", df, "mymodel.pkl")
    FP["scaler_GT"] = os.path.join("exp", "scaler", "GT", df, "GTmodel.pkl")

    FP["analyzeFile"] = os.path.join("exp", "Ori_model", "analyze_ori.txt")
    # FP["modelFp"] = os.path.join("exp", "Ori_model", "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
    FP["modelFp"] = os.path.join("exp", "Ori_model", "weights.hdf5")
    FP["logFp"] = os.path.join("exp", "Ori_model", "log.txt")
    return FP