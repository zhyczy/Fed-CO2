
def definite_version(version):
    
    if version == 46:
        print("Version46: make use of unlabeled test data")

    elif version == 47:
        print("Version47: Version46 + knowledge distillation")

    elif version == 48:
        print("Version48: Version46 + pure KL")

    elif version == 49:
        print("Version49: logits level distillation for Version47")

    elif version == 1:
        print("Version1: Introduce Shabby Adaptors")

    elif version == 2:
        print("Version2: Version1 Train Shabby adaptors independently")

    elif version == 3:
        print("Version3: Introduce Residual Adaptors")

    elif version == 4:
        print("Version4: Introduce Residual Adaptors independently")

    elif version == 15:
        print("Version15: Validate Version5 to check whether solely trained adaptor makes an impact on G and P branch")

    elif version == 16:
        print("Version16: Validate Version6 to check whether solely trained adaptor makes an impact on G and P branch")

    elif version == 7:
        print("Version7: V18 plus, generalize on adapted features based on Shabby Adaptors")

    elif version == 8:
        print("Version8: V7 plus generalize both on features and adapted features based on Shabby Adaptors")

    elif version == 9:
        print("Version9: V18 plus, generalize on adapted features based on Residual Adaptors")

    elif version == 10:
        print("Version10: V9 plus generalize both on features and adapted features based on Residual Adaptors")

    elif version == 23:
        print("Version23: Contrast version with Version 15, independent adaptor")

    elif version == 26:
        print("Version26: Contrast version with Version 15, independent adaptor and deep copy")

    elif version == 5:
            print("Version5: Updated Version2")

    elif version == 6:
        print("Version6: Updated Version4")

    elif version == 11:
        print("Version11: KL version of V5")

    elif version == 12:
        print("Version12: KL and CR for V5")

    elif version == 13:
        print("Version13: KL version of V6")

    elif version == 14:
        print("Version14: KL and CR for V6")

    elif version == 19:
        print("Version19: V7 + V5, Shabby Adaptors")

    elif version == 20:
        print("Version20: V7 plus + V5, Shabby Adaptors")

    elif version == 21:
        print("Version21: V9 + V6, Residual Adaptors")

    elif version == 22:
        print("Version22: V9 plus + V6, Residual Adaptors")

    elif version == 24:
        print("Version24: Contrast version with Version 15, deep personal classifier copy")

    elif version == 27:
        print("Version27: Find upper bound for new V5-V26")

    elif version == 29:
        print("Version29: Contrast version with Version 5, deep copy in test function")

    elif version == 30:
        print("Version30: Other Feature extractors adapt to this domain")

    elif version == 31:
        print("Version31: Add bias for Shabby Adaptor")

    elif version == 50:
        print("Version50: logits level distillation for Version48")

    elif version == 32:
        print("Version32: Seperate Training dataset and Testing dataset in two phases for version 50")

    elif version == 33:
        print("Version33: Remove communication in finetuning phase in Version 32")

    elif version == 34:
        print("Version34: version18 + Validation set judge, KD")

    elif version == 35:
        print("Version35: version34 + pretrained heads")

    elif version == 36:
        print("Version36: version18 + mutual learning, KD")

    elif version == 37:
        print("Version37: version36 + pretrained heads")

    elif version == 38:
        print("Version38: Simulate KDCL-Linear to get teacher logits for G and P branch")

    elif version == 39:
        print("Version39: Introduce KD weight for version38")

    elif version == 40:
        print("Version40: Generalize more on P branch with Shabby Adaptor")

    elif version == 41:
        print("Version41: Generalize more on P&G branch with Shabby Adaptor")

    elif version == 42:
        print("Version42: Generalize more on G branch with Shabby Adaptor")

    elif version == 43:
        print("Version43: Make series of trials on adaptor design based on version 15")
        print("bootle_neck residual design")

    elif version == 44:
        print("Version44: Add l2 regularization for P branch in version 18")

    elif version == 45:
        print("Version45: Add l1 regularization for P branch in version 18")

    elif version == 51:
        print("Version51: Add Orthogonal Constraint for G and P branch")

    elif version == 53:
        print("Version53: Add Orthogonal Constraint for G branch")

    elif version == 54:
        print("Version54: Add Orthogonal Constraint for P branch")

    elif version == 55:
        print("Version55: Add Orthogonal Constraint for P branch with pretrained heads")

    elif version == 56:
        print("Version56: Add Orthogonal branch in branch to boost G branch")

    elif version == 57:
        print("Version57: Generalize on backup branch and personal branch")

    elif version == 62:
        print("Check BN in version18, it seems we need to set spe_classifier eval() to keep the running mean and running var right")
        print("Now V18 has been modified to be the same as V62, add spe_classifier.eval()")

    elif version == 66:
        print("Version66: V18 + V65")

    elif version == 67:
        print("Version67: V63 and now classifier is only one layer")

    elif version == 68:
        print("version68: V65 + V63")

    elif version == 69:
        print("Version69: V66 + V63")