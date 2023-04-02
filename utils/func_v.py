def definite_version(version):
    if version == 1:
        print("Version1: Base model, two branches")

    elif version == 2:
        print("Version2: Third knowledge distillation branch-base version, reweighting is done with simple plus implementation")

    elif version == 3:
        print("Version3: version 2 + P branch generalized")

    elif version == 4:
        print("Version4: version 2 + G branch generalized")

    elif version == 5:
        print("Version5: version 2 + G&P generalized")

    elif version == 6:
        print("Version6: version 2 + G branch train")

    elif version == 7:
        print("Version7: version 4 + G branch train")

    elif version == 8:
        print("Version8: version 5 + G branch train")

    elif version == 9:
        print("Version9: version18 two phases, knowledge distillation G to P")

    elif version == 10:
        print("Version10: version9 two phases, G and P are trained seperately")

    elif version == 11:
        print("Version11: version10, knowledge distillation from P to G in second phase")

    elif version == 12:
        print("Version12: cmp version, fedbn add softmax")

    elif version == 13:
        print("Version13: cmp version, fedavg add softmax")

    elif version == 14:
        print("Version14: cmp version, remove BNs in classifier for Fedavg")

    elif version == 16:
        print("Version16: Remove BNs in C for version18")

    elif version == 19:
        print("Version19: Remove BNs in C for version32") 

    elif version == 15:
        print("Version15: Remove BNs in C for Version34")

    elif version == 48:
        print("Version48: Remove BNs in C for Version46")

    elif version == 49:
        print("Version49: Remove BNs in C for Version47")

    elif version == 20:
        print("Version20: cmp version, version18 add softmax")

    elif version == 21:
        print("Version21: Introduce hyper to produce parameters for classifier in P branch")

    elif version == 22:
        print("Version22: Third branch, Student model replaces adding aggregation, Aggregation model uses G branch still")

    elif version == 23:
        print("Version23: V22 Two phases, 1.train G&P 2.Distillation")

    elif version == 24:
        print("Version24: Check the influence of running mean and running var")

    elif version == 26:
        print("Version26: Check the influence of bn weight and bn bias")

    elif version == 27:
        print("Version27: Hyper now generates parameters for BN in P branch")

    elif version == 29:
        print("version29: comparison version, Hyper generates parameters for BN in P only")

    elif version == 30:
        print("version30: Hyper generates parameters for BN in P and G branch")

    elif version == 31:
        print("Version31: comparison version, Hyper generates parameters for BN in P&G only")

    elif version == 35:
        print("Version35: check the influence of running mean")

    elif version == 36:
        print("Version36: check the influence of running var")

    elif version == 39:
        print("Version39: aggregation, upper bound, g model accuracy")

    elif version == 40:
        print("Version40: aggregation, P head, approximation")

    elif version == 41:
        print("Version41: aggregation, G head, approximation")

    elif version == 45:
        print("Version45: Check the results after removing BNs for fedbn in classifier")

    elif version == 54:
        print("Version54: keep full BNs for v48")

    elif version == 55:
        print("Version55: keep full BNs for v49")

    elif version == 57:
        print("Version57: keep full BNs for V47")

    elif version == 50:
        print("Version50: learnable softmax for P&G")

    elif version == 51:
        print("Version51: train learnable softmax after aggregation")

    elif version == 52:
        print("Version52: train learnable softmax after aggregation as calibration")

    elif version == 53:
        print("Version53: train learnable softmax after aggregation as plus calibration")

    elif version == 58:
        print("Version58: sharpen aggregation weight for V39")

    elif version == 59:
        print("Version59: sharpen aggregation weight for V40")

    elif version == 60:
        print("Version60: sharpen aggregation weight for V41")

    elif version == 61:
        print("Version61: remove P branch for V39")

    elif version == 62:
        print("Version62: remove P branch for V41")

    elif version == 64:
        print("Version64: remove P branch for V58")

    elif version == 65:
        print("Version65: remove P branch for V60")